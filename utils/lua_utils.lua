-- This is a set of common functions in Lua
require 'lfs'
require 'pl'

local utils = {}

--------------------------------------------------------------------------------
-- SORT OF DEFAULTDICT
--------------------------------------------------------------------------------
-- Note that each element will get the same default value.
-- So if you make the default value a table,
-- each "empty" element in the returned table will effectively reference the same table.
-- If that's not what you want, you'll have to modify the function.
function utils.defaultdict(default)
    local tbl = {}
    local mtbl = {}
    mtbl.__index = function(tbl, key)
        local val = rawget(tbl, key)
        return val or default
    end
    setmetatable(tbl, mtbl)
    return tbl
end

--------------------------------------------------------------------------------
-- MAKE DIRECTORY IF IT DOESN'T EXIST
--------------------------------------------------------------------------------
function utils.make_dir_if_not_exists(dirpath)
    if not path.exists(dirpath) then
        lfs.mkdir(dirpath)
    end
end

--------------------------------------------------------------------------------
-- SPLIT STRING
--------------------------------------------------------------------------------
-- Taken from here: http://stackoverflow.com/questions/1426954/split-string-in-lua
-- I should probably use regex though
function utils.split(inputstr, sep)
    if sep == nil then
        sep = "%s"
    end
    local t={} ; i=1
    for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
        t[i] = str
        i = i + 1
    end
    return t
end

--------------------------------------------------------------------------------
-- READ FILE
--------------------------------------------------------------------------------
-- http://lua-users.org/wiki/FileInputOutput
-- see if the file exists
function utils.file_exists(file)
    local f = io.open(file, "rb")
        if f then f:close() end
    return f ~= nil
end

-- get all lines from a file, returns an empty 
-- list/table if the file does not exist
function utils.lines_from(filepath)
    if not path.exists(filepath) then return {} end
    lines = {}
    for line in io.lines(filepath) do 
        lines[#lines + 1] = line
    end
    return lines
end

-- Read cmd.csv, which is of form param,value
function utils.read_cmd_csv(filepath)
    local res = {}
    local lines = utils.lines_from(filepath)
    for i=1,#lines do 
        local parsed = utils.split(lines[i], ',')
        val = ''
        if #parsed == 2 then
            val = parsed[2]
        end
        res[parsed[1]] = val
    end
    return res
end

--------------------------------------------------------------------------------
-- SLICE TABLE
--------------------------------------------------------------------------------
function utils.subrange(t, first, last)
  local sub = {}
  for i=first,last do
      sub[#sub + 1] = t[i]
  end
  return sub
end

--------------------------------------------------------------------------------
-- SIZE OF TABLE
-- If table isn't indexed by number, can't do #t
--------------------------------------------------------------------------------
function utils.size_of_table(t)
    local count = 0
    for k,v in pairs(t) do
        count = count + 1
    end
    return count
end
--------------------------------------------------------------------------------
-- INTERLEAF TWO TABLES
--------------------------------------------------------------------------------
function utils.interleaf_tables(t1, t2)
    local interleafed = {}
    local i = 1
    local j = 1
    while (i <= #t1) and (j <= #t2) do
        table.insert(interleafed, t1[i])
        table.insert(interleafed, t2[j])
        i = i + 1
        j = j + 1
    end

    -- Potentially one of t1/t2 hasn't been completely appended if they aren't of the same length
    while i <= #t1 do
        table.insert(interleafed, t1[i])
        i = i + 1
    end
    while j <= #t2 do
        table.insert(interleafed, t2[j])
        j = j + 1
    end

    return interleafed
end

--------------------------------------------------------------------------------
-- CONVERT NON-INTEGER INDEXED TABLE TO INTEGER IN ORDER TO SAVE TO CSV
-- If table isn't indexed by number, csvigo can't do ipairs
--------------------------------------------------------------------------------
function utils.convert_table_for_csvigo(t)
    local new_table = {}
    for k,v in pairs(t) do
        if type(v) == 'boolean' then  -- Convert to 1 or 0
            if v then 
                v = 1
            else
                v = 0
            end
        end
        table.insert(new_table, {k, v})
        end
    return new_table
end

--------------------------------------------------------------------------------
-- INVERT TABLE: keys become values, values become keys
--------------------------------------------------------------------------------
function utils.invert_table(t)
    local new_table = {}
        for k,v in pairs(t) do
            new_table[v] = k
        end
    return new_table
end

--------------------------------------------------------------------------------
-- TERNARY
-- Example usage: local epochs = ternary_op(opt.use_google_model, 2, 3)
--------------------------------------------------------------------------------
function utils.ternary_op(condition, true_val, false_val)
    if condition then
        return true_val
    else
        return false_val
    end
end

--------------------------------------------------------------------------------
-- FUNCTIONAL
-- e.g. sum all items by: table.reduce(phoneme_nframes, function(a,b) return a+b end)
--------------------------------------------------------------------------------
table.reduce = function(list, fn) 
    local acc
    for k, v in ipairs(list) do
        if 1 == k then
            acc = v
        else
            acc = fn(acc, v)
        end 
    end 
    return acc 
end

function utils.map(func, tbl)
   local newtbl = {}
   for i,v in pairs(tbl) do
       newtbl[i] = func(v)
   end
   return newtbl
end

--------------------------------------------------------------------------------
-- ROUND VALUE
--------------------------------------------------------------------------------
function utils.round(x)
    if x%2 ~= 0.5 then
        return math.floor(x+0.5)
    end
    return x-0.5
end

return utils

