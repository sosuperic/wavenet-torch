-- Functions used for wavenet

local wavenet_utils = {}

----------------------------------------------------------------------------------------------------------------
-- Quantization: encoding and decoding
----------------------------------------------------------------------------------------------------------------
-- mu-law encoding: inputs must be in [-1,1], outputs also in [-1,1]
-- network will produce output in range [-1,1], which will then be quantized
-- In order to have all values in range [-1,1], divide by opt.max_val_for_quant and clamp
function wavenet_utils.quantize(wav, mu, max_val_for_quant)
    -- mu-law companding transformation
    wav = wav / max_val_for_quant
    wav:clamp(-1,1)
    wav = torch.cmul(torch.sign(wav), torch.log(1 + mu * torch.abs(wav)) / torch.log(1 + mu))

    -- quantize by placing values in bins from 1 to mu+1
    wav:apply(function (val)    -- get_bin(val)
        if val > 1 or val < -1 then
            print(val)
            print('Wav not properly standardized to be between [-1,1]')
            os.exit()
        end

        local num_bins = mu + 1
        local bin_width = 2 / (num_bins)
        if val == 1.0 then
            return num_bins
        else
            return math.floor((val + 1) / bin_width) + 1
        end
    end)

    return wav
end

-- Convert from bin number to value, mu-law expansion, and then scale
-- Value for bin is middle of that bin, e.g. 4 bins: leftmost-bin is [-1,0.5), value is -0.75
function wavenet_utils.decode(wav, mu, max_val_for_quant)
    -- Convert from bin number to value in [-1,1]
    wav:apply(function (bin)
        if bin < 1 or bin > mu + 1 then
            print(bin)
            print(string.format('Output bin number not between 1 and %d', mu + 1))
            os.exit()
        end

        local num_bins = mu + 1
        local bin_width = 2 / (num_bins)
        local val = (-1 + bin_width / 2) + (bin - 1) * bin_width
        return val
    end)
    -- mu-law expansion
    wav = torch.cmul(torch.sign(wav), (1/mu) * (torch.pow((1+mu), torch.abs(wav)) - 1))
    -- scale
    wav = wav * max_val_for_quant

    return wav
end

return wavenet_utils
