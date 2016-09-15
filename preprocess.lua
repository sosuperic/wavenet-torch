-- Functions that are run once to pre-process data

require 'pl'
require 'audio'
require 'csvigo'
local utils = require 'utils.lua_utils'

local preprocess = {}

------------------------------------------------------------------------------------------------------------------------
-- Params
------------------------------------------------------------------------------------------------------------------------
local PARAMS = {}

PARAMS['vctk'] = {}
PARAMS['vctk']['PERC_TRAIN'] = 0.98
PARAMS['vctk']['PERC_VALID'] = 0.01
PARAMS['vctk']['PERC_TEST'] = 0.01
PARAMS['vctk']['WAV_PATH'] = 'data/vctk/wav48/'
-- PARAMS['vctk']['OUT_PATH'] = 'data/processed/vctk/split_main' -- 0.8, 0.1, 0.1
PARAMS['vctk']['OUT_PATH'] = 'data/processed/vctk/split_98' -- 0.98, 0.01, 0.01

------------------------------------------------------------------------------------------------------------
-- Data splitting into train, valid, text
------------------------------------------------------------------------------------------------------------
function preprocess.write_split_to_file(tbl, fn)
    local f = io.open(fn, 'w')
    for i=1,#tbl do
        local speaker_id = tbl[i][1]
        local path = tbl[i][2]
        local num_samples = tbl[i][3]
        if i < #tbl then
            f:write(string.format('%s,%s,%s\n', speaker_id, path, num_samples))
        else
            f:write(string.format('%s,%s,%s', speaker_id, path, num_samples))
        end
    end
end

function preprocess.split_helper(perc_tr, perc_va, perc_te, wav_path, out_path)
    local tr, va, te = {}, {}, {}

    local subdirs = dir.getdirectories(wav_path)
    j = 0
    for _, subdir in ipairs(subdirs) do         -- each subdir is one person
        local wav_fps = dir.getfiles(subdir)
        local speaker_id = path.basename(subdir)
        speaker_id = speaker_id:sub(2,#speaker_id)
        for i, fp in ipairs(wav_fps) do
            if path.extension(fp) == '.wav' then
                local wav = audio.load(fp)
                print(j, fp)
                local num_samples = wav:size(1)
                local datum = {speaker_id, fp, num_samples}

                local ratio = i / #wav_fps
                if ratio <= perc_tr then table.insert(tr, datum)
                elseif (ratio > perc_tr) and (ratio <= perc_tr + perc_va) then table.insert(va, datum)
                else table.insert(te, datum)
                end

                j = j + 1
            end
        end
    end

    -- Save to files
    utils.make_dir_if_not_exists(out_path)
    preprocess.write_split_to_file(tr, path.join(out_path, 'train.txt'))
    preprocess.write_split_to_file(va, path.join(out_path, 'valid.txt'))
    preprocess.write_split_to_file(te, path.join(out_path, 'test.txt'))
end

function preprocess.split(opt)
    if opt.dataset == 'vctk' then
        preprocess.split_helper(
                PARAMS[opt.dataset]['PERC_TRAIN'],
                PARAMS[opt.dataset]['PERC_VALID'],
                PARAMS[opt.dataset]['PERC_TEST'],
                PARAMS[opt.dataset]['WAV_PATH'],
                PARAMS[opt.dataset]['OUT_PATH']
                )
    else
        print('Dataset must be vctk')
    end
end

------------------------------------------------------------------------------------------------------------
-- Get max value across all wavs in order to standardize values between [-1,1] for quantization
------------------------------------------------------------------------------------------------------------
function preprocess.get_max_wav_val_vctk()
    local subdirs = dir.getdirectories('data/vctk/wav48')
    local all_vals = {}
    local max = 0
    for _, subdir in ipairs(subdirs) do         -- each subdir is one person
        local wav_fps = dir.getfiles(subdir)
        print(subdir)
        for i, fp in ipairs(wav_fps) do
            if path.extension(fp) == '.wav' then
                local wav = audio.load(fp)
                local max_val = torch.abs(wav):max()
                if max_val > max then
                    max = max_val
                end
                table.insert(all_vals, max_val)
            end
        end
    end
    table.sort(all_vals)

    csvigo.save{path='data/processed/vctk/maxvals.csv', data=utils.convert_table_for_csvigo(all_vals)}
    print('Max: ' .. max)
end

------------------------------------------------------------------------------------------------------------------------
-- Main
------------------------------------------------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:option('-fn', '', 'split or get_max_wav_val_vctk')
cmd:option('-dataset', 'vctk', 'vcktk or ')
local opt = cmd:parse(arg)

if opt.fn == 'split' then
    preprocess.split(opt)
elseif opt.fn == 'get_max_wav_val_vctk' then
    preprocess.get_max_wav_val_vctk()
end
