-- -- to import all functions in a module, the syntax is:
-- import Data.List

-- -- to import select functions in a module, the syntax is:
import Data.List (nub, sort)

-- -- to import all functions in a module EXCEPT for a select few, the syntax is:
-- import Data.List hiding (num)

-- -- to import functions that might clash / shadow functions with the same name,
-- -- 	we can do a qualified import which will cause us to need to specify the
-- --	full namespace path to a function in order to use it
-- -- 		- e.g. to use the filter function from Data.Map, you would have to
-- --			do "Data.Map.filter"
-- import qualified Data.Map

-- -- a qualified import can be aliased to something a little more convenient to type
-- --	by using the following syntax.  Typing 'M' in place of 'Data.Map' would allow
-- --	us to access the Map module's filter function by just typing 'M.filter'
-- import qualified Data.Map as M


-- the documentation for Haskell's standard library and the list of modules: 
-- https://downloads.haskell.org/~ghc/latest/docs/html/libraries/


-- Search the Haskell search engine 'Hoogle' to find stuff.  You can search by
-- 	function name, module name or type signature:
-- https://hoogle.haskell.org/


numUniques :: (Eq a) => [a] -> Int
numUniques = length . nub





