-- Implement dataset interface for torchnet
-- Read from pre-processed splits produced by preprocess.lua -fn split

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

	self.n = #self.wav_lines
end

function WavenetDataset:size()
	return self.n
end

----------------------------------------------------------------------------------------------------------------
-- MODEL WITHOUT CONDITIONING
----------------------------------------------------------------------------------------------------------------
local NocondDataset, _ = torch.class('tnt.NocondDataset', 'tnt.WavenetDataset', tnt)

function NocondDataset:get(idx)
	-- local speakerid, wavpath, sampleidx = self:get_wavpath_and_sampleidx(idx)
	local speakerid, wavpath, _ = unpack(utils.split(self.wav_lines[idx], ','))

	-- Load wav and downsample, e.g. from 48000 to 16000
	local wav = audio.load(wavpath)		-- (numsamples_in_wav, 1)
	if wav:size(1) % self.downsample_factor ~= 0 then 	-- pad so we can reshape
		local pad_length = self.downsample_factor - (wav:size(1) % self.downsample_factor)
		wav = torch.cat(wav, torch.zeros(pad_length), 1)
	end
	wav = wav:reshape(wav:size(1) / self.downsample_factor, self.downsample_factor)
	wav = wav[{{},{1}}]				-- (seq_len, 1)

	local input = wavenet_utils.quantize(wav, self.mu, self.max_val_for_quant):transpose(1,2) -- (nsamples,1) -> (1,nsamples)
	local target = input:clone()									-- (1, nsamples)
	input = input:reshape(1,1,input:size(1), input:size(2))			-- Add 1st dimension for batch, 2nd dimension for feature maps
	
	return {
				speakerid = speakerid,
				input = input,
				target = target
			}
end
