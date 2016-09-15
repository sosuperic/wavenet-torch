-- For testing and debugging purposes

local test_suite = {}

------------------------------------------------------------------------------------------------------------------------
-- Test functions
------------------------------------------------------------------------------------------------------------------------
function test_suite.test_forward_pass()
    print('@@ TEST @@: Forward pass')
    local opt = {
        batchsize=32,num_expblocks=1,max_dilation=4,receptive_field_size=8,
        dilated_nummaps='4,4,4',onedim_nummaps=16,output_nummaps='8,8',
        mu=255,max_val_for_quant=2147418112,gpuids=''
    }
    local model = require 'model'
    model.init(opt)

    local net = model.get_nocond_net()
    local x = torch.rand(model.batchsize, 1, 1, model.receptive_field_size)
    local activations = net:forward(x)
    local crit = model.get_criterion()
    local targets = torch.ones(model.batchsize)
    local loss = crit:forward(activations, targets)
    print(string.format('Loss: %.4f', loss))
end

function test_suite.test_network()
    print('@@ TEST @@: Network')
    local opt = {
        mode='train',dataset='vctk',experiment='nocond',gpuids='',split_dirname='split_98',
        batchsize=32,num_expblocks=1,max_dilation=4,receptive_field_size=8,
        dilated_nummaps='4,4,4',onedim_nummaps=16,output_nummaps='8,8',
        mu=255,max_val_for_quant=2147418112,downsample_factor=3
    }
    local network = require 'network'
    network.tnt = require 'torchnet'
    local train_iterator = network:get_iterator('train', opt)
    for sample in train_iterator() do
        print(sample)
        break
    end
end

function test_suite.test_generate()
    print('@@ TEST @@: Generate sequence')
    local opt = {
        batchsize=32,num_expblocks=1,max_dilation=4,receptive_field_size=8,
        dilated_nummaps='4,4,4',onedim_nummaps=16,output_nummaps='8,8',
        mu=255,max_val_for_quant=2147418112
    }
    local wavenet_utils = require 'utils.wavenet_utils'
    local model = require 'model'
    model.init(opt)
    local net = model.get_nocond_net()

    -- Create initial input
    local x = torch.Tensor(1, 1, 1, opt.receptive_field_size)
    x = torch.zeros(1, 1, 1, opt.receptive_field_size)

    -- Create output sequentially
    outputs = {}
    local num_samples = 25
    for i=1,num_samples do
        local activations = net:forward(x)

        -- Get bin
        local _, bin = torch.max(activations, 2)
        bin = bin[1][1]

        -- Decode through inverse mu-law
        local output_val = wavenet_utils.decode(torch.Tensor({bin}), opt.mu, opt.max_val_for_quant)[1]

        -- Create next input by shifting and appending output
        x[{{1},{1},{1},{1, model.receptive_field_size - 1}}] = x[{{1},{1},{1},{2, opt.receptive_field_size}}]
        x[1][1][1][opt.receptive_field_size] = output_val
        table.insert(outputs, output_val)
    end
    print(outputs)
end

------------------------------------------------------------------------------------------------------------------------
-- Main
------------------------------------------------------------------------------------------------------------------------
test_suite.test_forward_pass()
test_suite.test_network()
test_suite.test_generate()
