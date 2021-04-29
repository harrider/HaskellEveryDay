-- function names in Haskell must begin with a lowercase letter
-- camelcase for function names



-- doubleMe function
doubleMe x = x + x


-- doubleUs function
doubleUs x y = x*2 + y*2

doubleUs' x y = doubleMe x + doubleMe y


-- doubleSmallNumber function
-- else part is required in Haskell
-- if/else is an expression; it always returns a value in Haskell
-- syntax is like a ternary in Python
doubleSmallNumber x = if x > 100 
                       then x 
                       else x*2

doubleSmallNumber' x = if x > 100 then x else x*2


-- doubleSmallNumberAddOne function
doubleSmallNumberAddOne x = (if x > 100 then x else x*2) + 1

doubleSmallNumberAddOne' x = (doubleSmallNumber x) + 1


-- conanO'Brien function
-- ' is a valid character for naming functions in Haskell
-- ' is used to denote striclty evaluated functions or slightly different versions of existing functions
conanO'Brien = "It's a-me, Conan O'Brien!"



-- List comprehension [ output_function | uboyt set, preducate ]
boomBangs xs = [ if x < 10 then "BOOM!" else "BANG!" | x <- xs, odd x ]


-- Custom length function using a list comprehension
length' xs = sum [1 | _ <- xs]


-- Function that uses a list comprehension to filters out lowercase characters
removeNonUppercase st = [ c | c <- st, c `elem` ['A'..'Z'] ]
