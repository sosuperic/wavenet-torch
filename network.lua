-- Create a network. Setup, move to GPU if needed, train and test

require 'optim'
require 'socket'
require 'pl'
require 'csvigo'
local utils = require 'utils.lua_utils'

local network = {}

function network:init(opt)
    self.tnt = require 'torchnet'
    self:setup_gpu(opt)
    if opt.mode == 'train' then
        self:setup_model(opt)
        self:setup_train_engine(opt)
    elseif opt.mode == 'test' then
        self:generate(opt)
    end
    self:move_to_gpu(opt)
end

------------------------------------------------------------------------------------------------
-- TRAINING
------------------------------------------------------------------------------------------------
function network:setup_model(opt)
    local method = {
        sgd = optim.sgd,
        adam = optim.adam,
        adagrad = optim.adagrad,
        adadelta = optim.adadelta,
        rmsprop = optim.rmsprop,
        adamax = optim.adamax
    }
    self.optim_method = method[opt.method]

    local split = utils.ternary_op(opt.train_on_valid, 'valid', 'train')
    local model = require 'model'
    model.init(opt)
    if opt.experiment == 'nocond' then
        self.nets = {model.get_nocond_net()}
    elseif opt.experiment == 'text' then
    elseif opt.experiment == 'textplusspeaker' then
    else
        print('experiment must be nocond, text, or textplusspeaker')
        os.exit()
    end

    self.criterions = {model.get_criterion()}
    self.train_iterator = self:get_iterator(split, opt)
    self.valid_iterator = self:get_iterator('valid', opt)
end

function network:setup_train_engine(opt)
    -- Set up engines and define hooks
    self.engine = self.tnt.OptimEngine()
    self.train_meter  = self.tnt.AverageValueMeter()
    self.valid_engine = self.tnt.SGDEngine()
    self.valid_meter = self.tnt.AverageValueMeter()
    self.engines = {self.engine, self.valid_engine}
    self.timer = self.tnt.TimeMeter{unit=true}

    -- Hooks for main (training) engine
    self.engine.hooks.onStartEpoch = function(state)
        self.train_meter:reset()
    end
    self.engine.hooks.onForwardCriterion = function(state)
        self.train_meter:add(state.criterion.output)
        print(string.format('Epoch: %d; avg. loss: %2.4f',
            state.epoch, self.train_meter:value()))
    end
    self.engine.hooks.onEndEpoch = function(state)
        -- Create directory to save models, etc. at end of first epoch
        -- Do it now to minimize chance of error and useless folder being created 
        if state.epoch == 1 and (not opt.dont_save) then
            self:make_save_directory(opt)
            self:save_opt(opt)
            self:setup_logger(opt)
        end

        -- Get loss on validation
        local valid_loss = math.huge
        if state.epoch % opt.eval_model_every_epoch == 0 then 
            print('Getting validation loss')
            self.valid_engine:test{
                network   = self.nets[1],
                iterator  = self.valid_iterator,
                criterion = self.criterions[1]
            }
            valid_loss = self.valid_meter:value()
        end

        if not opt.dont_save then
            local train_loss = self.train_meter:value()
            self.logger:add{train_loss, valid_loss, self.timer:value()}
        end

        -- Timer
        self.timer:incUnit()
        print(string.format('Avg time for one epoch: %.4f',
            self.timer:value()))

        -- Save model and loss
        if (state.epoch % opt.save_model_every_epoch == 0) and (not opt.dont_save) then
            local fn = string.format('net_e%d.t7', state.epoch)
            self:save_network(fn)
        end
    end
    self.engine.hooks.onEnd = function(state)
        if not opt.dont_save then
            local fn = string.format('net_e%d.t7', state.epoch)
            self:save_network(fn)
        end
    end

    -- Hooks for validation engine
    self.valid_engine.hooks.onStartEpoch = function(state)
        self.valid_meter:reset()
    end
    self.valid_engine.hooks.onForwardCriterion = function(state)
        self.valid_meter:add(state.criterion.output)
    end
    self.valid_engine.hooks.onEnd = function(state)
        print(string.format('Validation avg. loss: %2.4f',
            self.valid_meter:value()))
    end
end

function network:make_save_directory(opt)
    -- Create directory (if necessary) to save models to using current time 
    local cur_dt = os.date('*t', socket.gettime())
    local save_dirname = string.format('%d_%d_%d___%d_%d_%d',
        cur_dt.year, cur_dt.month, cur_dt.day,
        cur_dt.hour, cur_dt.min, cur_dt.sec)
    save_path = path.join(opt.models_dir, save_dirname)
    utils.make_dir_if_not_exists(save_path)
    self.save_path = save_path
end

function network:save_opt(opt)
    local fp = path.join(self.save_path, 'cmd')
    torch.save(fp .. '.t7', opt)
    csvigo.save{path=fp .. '.csv', data=utils.convert_table_for_csvigo(opt)}
end

function network:save_network(fn)
    local fp = path.join(self.save_path, fn)
    print(string.format('Saving model to: %s', fp))
    torch.save(fp, self.nets[1])
end


function network:setup_logger(opt)
    local fp = path.join(self.save_path, 'stats.log')
    self.logger = optim.Logger(fp)
    self.logger:setNames{'Train loss', 'Valid loss', 'Avg. epoch time'}
end

function network:train(opt)
    self.engine:train{
        network   = self.nets[1],
        iterator  = self.train_iterator,
        criterion = self.criterions[1],
        optimMethod = self.optim_method,
        config = {
            learningRate = opt.lr,
            learningRateDecay = opt.lr_decay,
            momentum = opt.mom,
            dampening = opt.damp,
            nesterov = opt.nesterov,
        },
        maxepoch  = opt.maxepochs,
    }
end

------------------------------------------------------------------------------------------------
-- TESTING
------------------------------------------------------------------------------------------------
function network:generate(opt)
    -- Load model and parameters used at train time
    require 'model'
    local model_path = path.join('models', opt.dataset, opt.experiment, opt.load_model_dir, opt.load_model_name)
    local cmd_path = path.join('models', opt.dataset, opt.experiment, opt.load_model_dir, 'cmd.csv')
    local traintime_opt = utils.read_cmd_csv(cmd_path)
    local net = torch.load(model_path)
    self.nets = {}
    self.criterions = {}
    self.engines = {}

    -- Create initial input
    local x = torch.Tensor(1, 1, 1, traintime_opt.receptive_field_size):zero()
    if opt.gpuids ~= '' then
        x = x:cuda()
    end

    -- Create output sequentially
    outputs = {}
    local num_samples = opt.gen_length
    local wavenet_utils = require 'utils.wavenet_utils'
    for i=1,num_samples do
        local activations = net:forward(x)
        -- Get bin
        local _, bin = torch.max(activations, 2)
        bin = bin[1][1]
        -- Decode through inverse mu-law
        local output_val = wavenet_utils.decode(torch.Tensor({bin}), traintime_opt.mu, traintime_opt.max_val_for_quant)[1]
        -- Create next input by shifting and appending output
        x[{{1},{1},{1},{1, traintime_opt.receptive_field_size - 1}}] = x[{{1},{1},{1},{2, traintime_opt.receptive_field_size}}]
        x[1][1][1][traintime_opt.receptive_field_size] = output_val
        table.insert(outputs, output_val)
    end
    print(outputs)
end

------------------------------------------------------------------------------------------------
-- USED BY BOTH TRAINING AND TESTING
------------------------------------------------------------------------------------------------
-- GPU ids are inversely mapped on Shannon for some reason. Then add 1 because lua is 1-based
function network:map_gpuid(id_or_tbl)
    if type(id_or_tbl) == 'number' then 
        return (3 - id_or_tbl) + 1
    else
        local mapped = {}
        for i, id in ipairs(id_or_tbl) do
            mapped[i] = (3 - id) + 1
        end
        return mapped
    end
end

function network:setup_gpu(opt)
    if opt.gpuids ~= '' then
        require 'cunn'
        require 'cutorch'
        -- require 'cudnn'
        if string.len(opt.gpuids) == 1 then
            cutorch.setDevice(self:map_gpuid(tonumber(opt.gpuids)))
            cutorch.manualSeed(123)
        end
        print(string.format('Using GPUs %s', opt.gpuids))
    end
end

function network:move_to_gpu(opt)
    if opt.gpuids ~= '' then
        for i,net in ipairs(self.nets) do
            if string.len(opt.gpuids) == 1 then
                net = net:cuda()
            else    -- multiple GPUs
                local gpus = utils.map(tonumber, utils.split(opt.gpuids, ','))
                gpus = self:map_gpuid(gpus)
                local dpt = nn.DataParallelTable(1, true, false)
                dpt:add(net, gpus)
                dpt.gradInput = nil
                dpt = dpt:cuda()
                self.nets[i] = dpt
            end
        end

        for i,criterion in ipairs(self.criterions) do
                -- print(cutorch.getDevice())
                criterion = criterion:cuda()
        end

        -- Copy sample to GPU buffer
        -- alternatively, this logic can be implemented via a TransformDataset
        -- local igpu, tgpu = cutorch.createCudaHostTensor(), torch.CudaTensor()
        local igpu, tgpu = torch.CudaTensor(), torch.CudaTensor()
        for i,engine in ipairs(self.engines) do
            engine.hooks.onSample = function(state)
                -- print(state)
                -- local igpu, tgpu = torch.CudaTensor(), torch.CudaTensor()
                -- print(cutorch.getDevice())
                igpu:resize(state.sample.input:size() ):copy(state.sample.input)
                tgpu:resize(state.sample.target:size()):copy(state.sample.target)
                state.sample.input  = igpu
                state.sample.target = tgpu
            end
        end
    end
end

function network:get_iterator(split, opt)
    return self.tnt.ParallelDatasetIterator{
        nthread = 1,
        closure = function()
        -- Closure's in separate threads, hence why we need to require torchnet
        -- Also reason (besides modularity) for putting dataset into separate file.
        -- Requiring packages at the start of this file won't be visible to this thread
            local tnt = require 'torchnet'
            require 'dataset'
            local dataset
            if opt.experiment == 'nocond' then
                dataset = tnt.NocondDataset(split, opt)
            elseif opt.experiment == 'text' or opt.experiment == 'textplusspeaker' then
                dataset = tnt.CondDataset(split, opt)
            end
            return dataset
        end,
    }
end

return network
