-- Implement dataset interface for torchnet
-- Read from pre-processed splits produced by preprocess.lua -fn split

--[[ 
Notes on methodology
--------------------
(This is not done in the most straightforward way in order to save on speed and memory)
Speed consideration: 
	- Each wav is some sequence of samples, where there are 48000 samples per second for the vctk dataset
	- One data point used for one forward-backward pass is a window over a wav, with the window size = receptive_field_size
	- The ith data point overlaps with (i+1)th input by (receptive_field_size - 1). I.e. it is offset by 1
	(This is for consecutive data points from the same wav)
	- Therfore, instead of having each get(idx) return one data point, and thus loading the wav file each time wastefully,
	we provide some batching here.
Memory consideration:
	- Storing say a self.idx2data table in __init, which would map the idx of the (i-th) point in the dataset
	to its wavpath and sampleidx requires too much memory
	- Thus, we have a function to compute the wavpath and the sampleidx

Illustration of methodology / definitions / terminology
-------------------------------------------------------
Each s is one sample.

wav1: [s1_1|s1_2|s1_3|s1_4|s1_5|s1_6|s1_7|s1_8|s9|s1_10|s1_11|s1_12|s1_13|s1_14|s1_15]
wav2: [s2_1|s2_2|s2_3|s2_4|s2_5|s2_6|s2_7|s2_8|s9|s2_10|s2_11|s2_12|s2_13|s2_14|s2_15|s2_16|s2_17]
wav3: [s3_1|s3_2|s3_3|s3_4|s3_5|s3_6|s3_7|s3_8|s9|s3_10|s3_11|s3_12]

For receptive_field_size=8, batchsize=4

1) When idx=1,
- get_wavpath_and_sampleidx(idx) returns {wav1path, 1}
- get(idx) returns batch1 created by combining:
	[s1_1|s1_2|s1_3|s1_4|s1_5|s1_6|s1_7|s1_8]
	[s1_2|s1_3|s1_4|s1_5|s1_6|s1_7|s1_8|s1_9]
	[s1_3|s1_4|s1_5|s1_6|s1_7|s1_8|s1_9|s1_10]
	[s1_4|s1_5|s1_6|s1_7|s1_8|s1_9|s1_10|s1_11]

2) When idx=2,
- get_wavpath_and_sampleidx(idx) returns {wav2path, 1}
	- (moves to wav2 because not enough samples to form batch)
	- Would need to stack 5:12, 6:13, 7:14, 8:15, plus 16 for the output
- get(idx) returns batch2 created by combining:
	[s2_1|s2_2|s2_3|s2_4|s2_5|s2_6|s2_7|s2_8]
	[s2_2|s2_3|s2_4|s2_5|s2_6|s2_7|s2_8|s2_9]
	[s2_3|s2_4|s2_5|s2_6|s2_7|s2_8|s2_9|s2_10]
	[s2_4|s2_5|s2_6|s2_7|s2_8|s2_9|s2_10|s1_11]

3) When idx=3,
- get_wavpath_and_sampleidx(idx) returns {wav2path, 5}
- get(idx) returns batch3 created by combining:
	[s2_5|s2_6|s2_7|s2_8|s2_9|s2_10|s2_11|s2_12]
	[s2_6|s2_7|s2_8|s2_9|s2_10|s2_11|s2_12|s2_13]
	[s2_7|s2_8|s2_9|s2_10|s2_11|s2_12|s2_13|s2_14]
	[s2_8|s2_9|s2_10|s2_11|s2_12|s2_13|s2_14|s1_15]
--]]

require 'pl'
require 'audio'
local utils = require 'utils.lua_utils'
local wavenet_utils = require 'utils.wavenet_utils'
local tnt = require 'torchnet'

---------------------------------------------------------------------------------------------------------------
-- Datasets, Parameters, Paths
----------------------------------------------------------------------------------------------------------------
local PATHS = {}
PATHS['vctk'] = {}
PATHS['vctk']['SPLIT_PATH'] = 'data/processed/vctk/'

----------------------------------------------------------------------------------------------------------------
-- WAVENET MODEL (Parent class for models with and without conditioning)
----------------------------------------------------------------------------------------------------------------
local WavenetDataset, _ = torch.class('tnt.WavenetDataset', 'tnt.Dataset', tnt)

function WavenetDataset:__init(split, opt)
	self.batchsize = opt.batchsize
	self.downsample_factor = opt.downsample_factor
	self.mu = opt.mu
	self.max_val_for_quant = opt.max_val_for_quant
	self.receptive_field_size = opt.receptive_field_size
	self.wav_lines = utils.lines_from(path.join(PATHS[opt.dataset]['SPLIT_PATH'], opt.split_dirname, split .. '.txt'))

	self.n = 0
	for i=1,#self.wav_lines do
		local parsed = utils.split(self.wav_lines[i], ',')
		local wav_numsamples = tonumber(parsed[3])

		-- Get downsampled amount
		wav_numsamples = math.ceil(wav_numsamples / self.downsample_factor)
		local num_batches = self:calc_num_batches_in_wav(wav_numsamples)
		self. n = self.n + num_batches

		self.wav_lines[i] = {tonumber(parsed[1]), parsed[2], wav_numsamples}
	end
end
function WavenetDataset:size()
	return self.n
end

----------------------------------------------------------------------------------------------------------------
-- Retrieving batches
----------------------------------------------------------------------------------------------------------------
function WavenetDataset:calc_num_batches_in_wav(wav_numsamples)
	-- Formula: + 1 for offset, -1 for extra sample needed for output
	-- In above example for wav1 (assuming example shows already downsampled wavs),
		-- math.floor((15 - 8 + 1 - 1) / 4) = 1 (short 1 sample needed for output)
		-- If 16 samples instead, math.floor((16 - 8 + 1 - 1) / 4) = 2
	return math.floor((wav_numsamples - self.receptive_field_size + 1 - 1) / self.batchsize)
end

function WavenetDataset:get_wavpath_and_sampleidx(idx)
	-- Methodology: just iterate over batches while updating pointers until cur_idx == idx
	local cur_idx = 1							-- idx of current batch out of entire dataset
	local cur_wav_idx = 1						-- idx of current wav
	local cur_wav_sampleidx = 1					-- idx of sample in current wav
	local cur_wav_numsamples = self.wav_lines[cur_wav_idx][3] 	-- num samples in current wav (already downsampled)

	while cur_idx ~= idx do 					-- move pointers forward
		-- Case 1: batch is not in current wav
		local cur_wav_numbatches = self:calc_num_batches_in_wav(cur_wav_numsamples)
		if idx > cur_idx + cur_wav_numbatches - 1 then
			cur_idx = cur_idx + cur_wav_numbatches
			cur_wav_idx = cur_wav_idx + 1
			cur_wav_numsamples = self.wav_lines[cur_wav_idx][3]
		-- Case 2: batch is in wav, just move the sampleidx forward
		else
			cur_idx = cur_idx + 1
			cur_wav_sampleidx = cur_wav_sampleidx + self.batchsize
		end
	end
	local cur_wav_speakerid = self.wav_lines[cur_wav_idx][1]
	local cur_wav_path = self.wav_lines[cur_wav_idx][2]
	-- print(cur_wav_path, idx, cur_wav_sampleidx, cur_wav_numsamples)
	return cur_wav_speakerid, cur_wav_path, cur_wav_sampleidx
end

----------------------------------------------------------------------------------------------------------------
-- MODEL WITHOUT CONDITIONING
----------------------------------------------------------------------------------------------------------------
local NocondDataset, _ = torch.class('tnt.NocondDataset', 'tnt.WavenetDataset', tnt)

function NocondDataset:get(idx)
	local speakerid, wavpath, sampleidx = self:get_wavpath_and_sampleidx(idx)

	-- Load wav and downsample, e.g. from 48000 to 16000
	local wav = audio.load(wavpath)		-- (numsamples_in_wav, 1)
	if wav:size(1) % self.downsample_factor ~= 0 then 	-- pad so we can reshape
		local pad_length = self.downsample_factor - (wav:size(1) % self.downsample_factor)
		wav = torch.cat(wav, torch.zeros(pad_length), 1)
	end
	wav = wav:reshape(wav:size(1) / self.downsample_factor, self.downsample_factor)
	wav = wav[{{},{1}}]

	-- Get batched inputs and targets
	local batched_inputs = torch.zeros(self.batchsize, 1, 1, self.receptive_field_size)
	local batched_targets = {}
	local j = 1
	for i=sampleidx, sampleidx + self.batchsize - 1 do
		-- Get one data point's input and target
		local input = wav[{{i, i + self.receptive_field_size - 1}, {}}]	-- (RECEPTIVE_FIELD_SIZE,1)
		local targetidx = i + self.receptive_field_size
		local target = wav[targetidx][1]

		-- Quantize input (target gets quantized after it's converted to tensor)
		input = wavenet_utils.quantize(input, self.mu, self.max_val_for_quant)

		-- Manipulate shape so that one input is (1,1,RECEPTIVE_FIELD_SIZE), as expected by model
		input = input:reshape(1, 1, self.receptive_field_size)

		-- Add to batch
		batched_inputs[j] = input
		table.insert(batched_targets, target)
		j = j + 1
	end
	batched_targets = torch.Tensor(batched_targets)
	batched_targets = wavenet_utils.quantize(batched_targets, self.mu, self.max_val_for_quant)

	return {
				speakerid = speakerid,
				input = batched_inputs,
				target = batched_targets
			}
end
