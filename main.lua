-- Main file to train network
require 'pl'
require 'utils.lua_utils'

---------------------------------------------------------------------------
-- Params
---------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Example:')

-- th main.lua -output_nummaps 32,16 -num_expblocks 3 -max_dilation 256 -dilated_nummaps 16,16,16,16,16,16,16,16,16 -lr 0.01 -method sgd -gpuids 0 -split_dirname split_oneperson -save_model_every_epoch 1 -eval_model_every_epoch 3 -train_on_valid -onedim_nummaps 32
-- th main.lua -mode test -load_model_dir 2016_9_20___2_12_52 -load_model_name net_e2.t7 -gpuids 3

cmd:text('Options:')
-- Training / Testing
cmd:option('-mode', 'train', 'train or test')
cmd:option('-dataset', 'vctk', 'vctk or')
cmd:option('-experiment', 'nocond', 'nocond / text / textplusspeaker')
cmd:option('-gpuids', '', "e.g. '0 or 2,3")
cmd:option('-train_on_valid', false, 'Train on valid instead of training. Used to debug because it is faster')
cmd:option('-batchsize', 32, 'number of examples in minibatch')
cmd:option('-maxepochs', 100, 'max number of epochs to train for')
cmd:option('-downsample_factor', 3, '48000 samples per second -> 16000 samples per second')
cmd:option('-mu', 255, 'quantization')
cmd:option('-load_model_dir', '', 'directory name from which to load model')
cmd:option('-load_model_name', '', 'e.g. net_e3.t7')
cmd:option('-gen_length', 100, 'number of samples to generate')
-- Model size
cmd:option('-num_expblocks', 1, 'number of exponentially dilated blocks')
-- cmd:option('-max_dilation', 4, 'max dilation in one block , e.g. 1, 2, 4, 8, ...')
-- cmd:option('-dilated_nummaps', '16,16,16', 'num maps for dilated convs in one expblock; need n, where 2^(n-1) = max_dilation')
cmd:option('-max_dilation', 256, 'max dilation in one block , e.g. 1, 2, 4, 8, ...')
cmd:option('-dilated_nummaps', '16,16,16,16,16,16,16,16,16', 'num maps for dilated convs in one expblock; need n, where 2^(n-1) = max_dilation')
cmd:option('-onedim_nummaps', 32, 'num maps for 1x1 convolutions in res block')
cmd:option('-output_nummaps', 16, 'number of feature maps for 1x1 convs in output')
-- cmd:option('-output_nummaps', '32,16', 'number of feature maps for 1x1 convs in output')
-- Optimization
cmd:option('-method','sgd', 'which optimization method to use')
cmd:option('-lr', 1e-2, 'learning rate')
cmd:option('-lr_decay', 0, 'learning rate decay')
cmd:option('-mom', 0, 'momentum')
cmd:option('-damp', 0, 'dampening')
cmd:option('-nesterov', false, 'Nesterov momentum')
-- Bookkeeping
cmd:option('-split_dirname', 'split_98', 'name of directory containing splits')
cmd:option('-dont_save', false, 'Save or not. Use true for testing / debugging')
cmd:option('-save_model_every_epoch', 1, 'how often to save model')
cmd:option('-eval_model_every_epoch', 5, 'how often to eval model on validation set')
cmd:option('-notes', '', 'String of notes, e.g. using batch norm. To keep track of iterative testing / small modifications')
local opt = cmd:parse(arg)

-- Calculate some things
opt.models_dir = path.join('models', opt.dataset, opt.experiment)
opt.save_test_dir = path.join('outputs', opt.dataset, opt.experiment)
opt.receptive_field_size = 2*opt.max_dilation + (opt.num_expblocks-1) * (2*opt.max_dilation - 1)   -- this is also the input size

-- Also not really a flag. Value calculated by going through wavs and getting max value. This is used to quantize.
opt.max_val_for_quant = 2147418112

---------------------------------------------------------------------------
-- Training
---------------------------------------------------------------------------
local network = require 'network'
network:init(opt)

if opt.mode == 'train' then
    network:train(opt)
elseif opt.mode == 'test' then
end
