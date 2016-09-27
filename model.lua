-- The wavenet model
-- rf := receptive_field_size 

--[[
Note on causal convolutions and onedim_nummaps (number of feature maps for 1x1 convolution).

The output of a causal dilated convolution has the same size as the input. This is done by filling with the
input, as shown in figure 3. However, the causal convolution can change the number of feature maps (controlled
by dilated_nummaps). Thus, when filling, we must match the number of feature maps in the input to the number
of feature maps in the output of the convolution. There are two cases:
1) The very first causal convolution. Here, the number of input feature maps is 1. We match by replicating
along that dimension.
2) Not the first causal convolution. Here, the number of input feature maps is determined by the output
of the 1x1 convolution. This is set by onedim_nummaps. We match by passing the input through a 1x1 convolution.

This number-of-feature-maps matching also occurs in the residual connection; this is the reason for
the first_conv parameter in get_convandfill_block and get_res_block
--]]

require 'nn'
require 'rnn'
require 'nngraph'

utils = require 'utils.lua_utils'

local model = {}

-- Get dilated convolution
function model.get_convandfill_block(first_conv, dilation, onedim_nummaps, dilated_nummaps)
    -- Left side, get neurons from input to fill result of convolution
    -- local fill_len = dilation
    local get_fill = nn.Sequential()
    get_fill:add(nn.Replicate(2))       -- (1,1 or onedim_nummaps,1,seq_len) -> (2,1,1 or onedim,1,seq_len)
    get_fill:add(nn.SplitTable(1,5))    -- table of (batch,1 or onedim_nummaps,1,seq_len)
    
    local get_fill_select = nn.Sequential()
    get_fill_select:add(nn.Narrow(4,1,dilation))    -- (batch,1 or onedim_nummaps,1,dilation)
    if first_conv then
        get_fill_select:add(nn.Squeeze(2))                      -- (batch,1,dilation)
        get_fill_select:add(nn.Replicate(dilated_nummaps, 2))   -- (batch,dilated_nummaps,1,dilation)
    else
        get_fill_select:add(nn.SpatialConvolution(onedim_nummaps, dilated_nummaps, 1, 1, 1, 1, 0, 0))
    end

    local get_fill_mask = nn.Sequential()
    get_fill_mask:add(nn.MulConstant(0, false))             -- (batch,1 or onedim_nummaps,1,seq_len)
    get_fill_mask:add(nn.Narrow(4,dilation+1,-1))           -- (batch,1 or onedim_nummaps,1,seq_len-dilation)
    if first_conv then
        get_fill_mask:add(nn.Squeeze(2))                    -- (batch,1,seq_len-dilation)
    else                                                    -- (batch,onedim_nummaps,1,seq_len-dilation)
        get_fill_mask:add(nn.Select(2,1))                   -- (batch,1,seq_len-dilation)
    end
    get_fill_mask:add(nn.Replicate(dilated_nummaps, 2))     -- (batch,dilated_nummaps,1,seq_len-dilation)
    
    get_fill:add(nn.ParallelTable():add(get_fill_select):add(get_fill_mask))
    get_fill:add(nn.JoinTable(4, 4))                -- (1,1 or onedim_nummaps,1,seq_len) (join on 4th dim, 4 input dims)

    -- -- Right side, convolve and pad
    local conv_and_pad = nn.Sequential()
    if first_conv then
        conv_and_pad:add(nn.SpatialDilatedConvolution(1, dilated_nummaps, 2, 1, 1, 1, 0, 0, dilation, 1))
    else
        conv_and_pad:add(nn.SpatialDilatedConvolution(onedim_nummaps, dilated_nummaps, 2, 1, 1, 1, 0, 0, dilation, 1))
    end
    conv_and_pad:add(nn.Padding(4, -1 * dilation)) -- (batch,dilated_nummaps,1,rf-fill_length) -> (batch,dilated_nummaps,1,rf)

    -- -- Combine fill and conv, add to block
    local block = nn.Sequential()
    local conv_and_fill = nn.ConcatTable():add(get_fill):add(conv_and_pad)  -- table of (batch,dilated_nummaps,1,rf)
    block:add(conv_and_fill)
    block:add(nn.CAddTable())                                               -- (batch,dilated_nummaps,1,rf)

    -- return get_fill
    return block
end

-- Get residual block: nngraph's output is a table of the skip_path and the straight_path
function model.get_res_block(input, first_conv, dilation, dilated_nummaps, onedim_nummaps)
    -- Skip path
    local skip_path = nn.Sequential()
    skip_path:add(nn.Identity())                        -- (batch,1 or onedim_nummaps,1,rf)
    if first_conv then                                  -- (batch,1,1,rf) -> (batch,onedim_nummaps,1,rf)
        skip_path:add(nn.Squeeze(2))                    -- (batch,1,rf)
        skip_path:add(nn.Replicate(onedim_nummaps, 2))  -- (batch,onedim_nummaps,1,rf)
    end

    -- Straight path (path with layers)
    local straight_path = nn.Sequential()

    -- Straight path: convolutions with gated activation units
    -- Get filter and gate
    local filter = nn.Sequential()
    filter:add(model.get_convandfill_block(first_conv, dilation, onedim_nummaps, dilated_nummaps))
    filter:add(nn.Tanh())
    local gate = nn.Sequential()
    gate:add(model.get_convandfill_block(first_conv, dilation, onedim_nummaps, dilated_nummaps))
    gate:add(nn.Sigmoid())
    -- Add to straight path
    straight_path:add(nn.ConcatTable():add(filter):add(gate))
    straight_path:add(nn.CMulTable())

    -- Straight path: 1 x 1 Convolution
    straight_path:add(nn.SpatialConvolution(dilated_nummaps, onedim_nummaps, 1, 1, 1, 1, 0, 0)) -- (batch,onedim_nummaps,1,rf)

    -- Residual block: Glue together paths
    local res_block = nn.Sequential()
    local paths = nn.ConcatTable():add(skip_path):add(straight_path)
    res_block:add(paths)
    res_block = res_block(input)

    return res_block
end

-- Get stacked residual blocks, where dilation increases exponentially
function model.get_expdilated_res_block(input, first_block, not_last_block, dilations, dilated_nummaps_tbl, onedim_nummaps)
    local results = {input}
    local skips = {}
    for i, dil in ipairs(dilations) do
        local first_conv = first_block and i == 1
        local res_block = model.get_res_block(results[i], first_conv, dil, dilated_nummaps_tbl[i], onedim_nummaps)

        res_block_skip = nn.SelectTable(2)(res_block)
        table.insert(skips, res_block_skip)

        if i ~= #dilations or not_last_block then   -- result not needed for last block, nngraph will throw error for unused node
            res_block_result = nn.Sequential():add(nn.CAddTable())
            res_block_result = res_block_result(res_block)
            table.insert(results, res_block_result)
        end
    end
    return results, skips
end

-- Combine intermediate skip paths to produce output
-- Second 1x1 convolution reduces size to 256 in order to perform softmax
function model.get_output_from_skips(skips, onedim_nummaps, output_nummaps)
    local output = nn.Sequential()
    output:add(nn.Identity())
    output:add(nn.CAddTable())              -- (batch,onedim_nummaps,1,seq_len)
    output:add(nn.ReLU())
    output:add(nn.SpatialConvolution(onedim_nummaps, output_nummaps, 1, 1, 1, 1, 0, 0))     -- (batch,output_nummaps,1,seq_len)
    output:add(nn.ReLU())

    output:add(nn.SpatialConvolution(output_nummaps, model.mu + 1, 1, 1, 1, 1, 0, 0))       -- (batch,256,1,seq_len)
    output:add(nn.Squeeze(3))                                                               -- (batch,256,seq_len)
    output:add(nn.ReLU())
    output:add(nn.Transpose({2,3}))                                                         -- (batch,seq_len,256)
    output:add(nn.SplitTable(1,3))                                                          -- table of (seq_len,256)
    local map = nn.Sequential():add(nn.LogSoftMax()):add(nn.Unsqueeze(1))
    output:add(nn.MapTable():add(map))                                                      -- table of (seq_len,256) -> table of (1,seq_len,256)
    output:add(nn.JoinTable(1,3))                                                           -- (batch,seq_len,256)

    output = output(skips)
    return output
end

function model.flatten_tables(tbl_of_tbls)
    local flattened = {}
    for i, tbl in ipairs(tbl_of_tbls) do
        for j, item in ipairs(tbl) do
            table.insert(flattened, item)
        end
    end
    return flattened
end

-- Get table of exponentially increasing dilation values for one block
function model.get_dilations(max_dilation)
    local dilation = max_dilation
    local dilations = {}
    while dilation >= 1 do
        table.insert(dilations, 1, dilation)
        dilation = math.floor(dilation / 2)
    end
    return dilations
end

function model.get_nocond_net()
    local input = nn.Identity()()
    local first_conv = nn.Identity()
    local prev = {input}
    local skips = {}
    for i=1,model.num_expblocks do
        local first_block = i == 1
        local not_last_block = i ~= model.num_expblocks
        expblock_results, expblock_skips = model.get_expdilated_res_block(
            prev[#prev], first_block, not_last_block,
            model.dilations, model.dilated_nummaps, model.onedim_nummaps)
        prev = expblock_results
        table.insert(skips, expblock_skips)
    end
    local all_skips = model.flatten_tables(skips)
    local output = model.get_output_from_skips(all_skips, model.onedim_nummaps, model.output_nummaps)
    local net = nn.gModule({input}, {output})

    return net
end

function model.get_criterion()
    return nn.SequencerCriterion(nn.ClassNLLCriterion(), true)
end

function model.init(opt)
    model.num_expblocks = opt.num_expblocks
    model.max_dilation = opt.max_dilation
    model.dilations = model.get_dilations(opt.max_dilation)
    model.receptive_field_size = opt.receptive_field_size
    model.mu = opt.mu

    model.dilated_nummaps = utils.map(tonumber, utils.split(opt.dilated_nummaps, ','))
    model.onedim_nummaps = opt.onedim_nummaps
    model.output_nummaps = opt.output_nummaps
end

return model
