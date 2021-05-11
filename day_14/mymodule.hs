-- -- to import all functions in a module, the syntax is:
-- import Data.List

-- -- to import select functions in a module, the syntax is:
import Data.List (nub, sort)
import qualified Data.Char
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



-- ===========================================================================================
-- My first implementation of a function in Haskell where I 100% implemented it myself :D
findIndex' :: (Eq a) => Int -> a -> [a] -> Maybe Int
findIndex' _ _ [] = Nothing
findIndex' i t [x]
  | x == t = Just i
  | otherwise = Nothing
findIndex' i t (x:xs)
  | x == t = Just i
  | otherwise = findIndex' (i+1) t xs
-- ===========================================================================================


-- the 'ord' and 'chr' functions convert characters to their corresponding numbers and vice versa
--
-- the difference between the 'ord' values of 2 characters is equal to how far apart they are in the
-- 	Unicode table
--
-- in order to create a simple Caesar cipher, we first create conver the string input into a list
-- 	of numbers.  Then we add the shift amount to each number.  Then we convert each number
-- 	back into a character.  And since a String is an array of characters, we end up rebuilding
-- 	a string with shifted characters.
encode :: Int -> String -> String
encode shift msg =
  let ords = map Data.Char.ord msg
      shifted = map (+ shift) ords
  in map Data.Char.chr shifted


-- To decode a shifted message, we just take the original shifted value, negate it and then apply
-- 	the negated shifted value to the encode function.  This inturn, undoes the shifting that 
-- 	the origina encode function did.
decode :: Int -> String -> String
decode shift msg = encode (negate shift) msg



-- Association lists (also called 'dictionaries') ae lists that are used to store key-value pairs
-- 	where ordering doesn't matter.
