-- Notes
-- =======================================================================
-- function names in Haskell must begin with a lowercase letter

-- camelcase for function names

-- everything has a type, even functions:  Explicity types start with a capital letter (ie Char, Bool)
-- 	nameOfThing :: <Type>   (this reads like, "nameOfThing has type of <type>"

-- if a function is comprised only of special characters, it's considered to be an infix function
-- 	by default (ie. +, -, /, *, ==, /=, etc...)

-- 	- You must surround an infix function in parenthesis if you ant to:
-- 		- examine it's type ... :t (+)
-- 		- pass it to another function ... myFunction (-) <arg2>
-- 		- call it as a prefix function ... (+) 1 1   (ie. 1 + 1)



-- =======================================================================



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
-- Functions have their own types based on their signatures
-- [Char] is synonymous with String, so it's clearer to just write String
-- removeNonUpperCase :: [Char] -> [Char]
removeNonUpperCase :: String -> String
removeNonUpperCase st = [ c | c <- st, c `elem` ['A'..'Z'] ]


-- The size and types contained in a tuple determine its type
triangles = [ (a,b,c) | c <- [1..10], b <- [1..c], a <- [1..b], a^2 + b^2 == c^2 ]


-- Writing out hte signature of a function that takes several arguments
addThree :: Int -> Int -> Int -> Int
addThree x y z = x + y + z


-- Overview of common types
--
-- Int is bounded and signed, -2147483647 <= Int <= 2147483647
-- Integer is unbounded and signed, -Inf <= Integer <= Inf
factorial :: Integer -> Integer
factorial n = product [1..n]


-- Float is a real floating point with single precision (32-bit :: 9-bit . 23-bit)
circumference :: Float -> Float
circumference r = 2 * pi * r


-- Double is a real floating point with double precision (64-bit :: 12-bit . 52-bit)
circumference' :: Double -> Double
circumference' r = 2 * pi * r


-- Bool is a boolean and has one of 2 values: True or False
isHotDog :: String -> Bool
isHotDog x = x == "Hot Dog"


-- Char represents a character and is surrounded by '' (single quotes)
-- a list of Char (ie [Char] ) is a String


-- Functions that have "type variables" (ie. types like [a] , (a,b,c) ) are called
-- 	polymorphic functions
myFst' :: (a,b) -> a
myFst' t = fst t

mySnd' :: (a,b) -> b
mySnd' t = snd t


-- Typeclasses are a sort of interface that define some behavior.

-- If a type is a part of a type class (ie. (Eq a) , then it's basically saying that:
-- 	type variable 'a' implements the behavior 'Eq'
--
-- 	- (Eq a) => a -> a -> Bool would read like:
--		a must be a member of typeclass Eq
--
--	- the Eq typeclass provides an interface for testing for equality

--	- All standard Haskell types (except for IO) and functions are a part of 
--		the Eq typeclass

--	- All of the types covered so far except for functions are a part of the
--		Ord typeclass
--
--	- All of the types covered so far except for functions are a part of the
--		Show typeclass
--		- The most used function that deals with the Show typeclass is 'show'
--			- 'show' takes a value whose type is a member of Show and presents
--				it as a string
--			- ex) show 3  ... output: "3"
--			- ex) show True ... output: "True"
--			- ex) show 3.14 ... output: "3.14"

--	- The Read typeclass is kind of the opposite of Show
--		- the 'read' function takes a string and returns a type that's a member of Read
--			- ex) read "3" ... output: 3
--			- ex) read "True" ... output: True
--			- ex) read "3.14" ... output: 3.14
--
--		- if you try to use the 'read' function by itself, you will get an error
--			because Haskell doesn't know what type the value should be cast into
--
--			- ex) read "4" ... output: error
--			- ex) read "4" + 3 ... output: 7  (because the compiler can infer the type)
--			- eX) read "4" :: Int ... output: 4  (because we used a type annotation to
--				tell the compiler what type to cast to)

-- In a function delcaration, everyting before a '=>' is called a class constraint

-- Some basic typeclasses are:
--
-- Eq :: is used for types that support equality testing
myEq' :: (Eq a) => a -> a -> Bool
myEq' x y = x == y

-- Ord :: is used for types that have an ordering
myGt :: (Ord a) => a -> a-> Bool
myGt x y = x > y

-- Show :: is used for types that can be presented as strings
myToString :: (Show a) => a -> String
myToString s = show s

-- Read :: is used for types that are presented as strings that can be parsed into another type
myParse :: (Read a) => String -> a
myParse s = read s

-- Enum :: members are sequentially ordered types - they can be enumerated
-- 	- The main advantage of the Enum typeclass is that its types can be used in list ranges
--		- ex) ['a'..'e'] ... output: ['a', 'b', 'c',  'd', 'e']
--		- ex) [LT..GT] ... output: [LT, EQ, GT]
--		- ex) succ 'B' ... output: 'C'
--
--	- They also have defined successors and predecessors that can be obtained by using:
--		- succ :: funciton to get the successor
--		- pred :: function to get the predecessor
--
--	- Types in this class include: (), Bool, Char, Ordering, Int, Integer, Float and Double
getSuccessor :: (Enum a) => a -> a
getSuccessor x = succ x

getPredecessor :: (Enum a) => a -> a
getPredecessor x = pred x


-- Bounded :: members have an upper and lower bound
-- 	- the minBound and maxBound function have a type of (Bounded a) => a
-- 		- they're essentially polymorphic constants
-- 		- they are  used by calling the function with a type annotation
-- 			
-- 			- ex) minBound :: Int ... output: -2147483648
-- 			- ex) maxBound :: Char ... output: '\1114111'
-- 			- ex) minBound :: Bool ... outptut: False
-- 			- ex) maxBound :: Bool ... output: True


-- Num :: is a numeric typeclass whose members have a property of being able to act like numbers
-- 	- Includes all numbers, including real and integral numbers
-- 	- whole numbers are polymorphic constants because they can act like any type that's a member of
-- 		the 'Num' typeclass
-- 		
-- 		- ex) 20 :: Int ... output: 20
-- 		- ex) 20 :: Integer ... output: 20
-- 		- ex) 20 :: Float ... output: 20.0
-- 		- ex) 20 :: Double ... output: 20.0


-- Integral :: is also a numeric typeclass, but it only includes integral (whole) numbers
-- 	- In this typeclass are: Int and Integer


-- Floating :: is also a numeric typeclass, but it only includes floating point numbers
-- 	- In this typeclass are: Float and Double


-- NOTE: fromIntegral is a very useful function for dealing with numbers
-- 	- its type declaration looks like:
-- 		fromIntegral :: (Num b, Integral a) => a -> b
--
-- 	- It takes a value of type Int or Integer and returns a more general value of typeclass Num
-- 	-
-- 	- This essentially helps you convert from Int / Integer to a general type that can be used
-- 		with non Integral types such as Float or Double
--
-- 		- ex) length [1,2,3] + 3.2 ... output: error 
-- 			- becasue (+) takes 2 arguments of the same type but we would be 
-- 				trying to add 3 :: Int + 3.2 :: Float
--
-- 		- ex) fromIntegral (length [1,2,3]) + 3.2 ... output: 6.2 
-- 			- because fromIntegral would turn the length Int into a Num and the
-- 				compiler could cast the Num into a Float and add it to 3.2


-- Pattern Matcing in Haskell allows you to define separate function bodies for different patterns
lucky :: (Integral a) => a -> String
lucky 7 = "LUCKY NUMBER SEVEN!"
lucky x = "Sorry, you're out of luck, pal!"


-- Using pattern matching instead of if/else statements allows for more succint code that can deal
-- 	with a larger set of runetime cases
sayMe :: (Integral a) => a -> String
sayMe 1 = "One!"
sayMe 2 = "Two!"
sayMe 3 = "Three!"
sayMe 4 = "Four!"
sayMe 5 = "Five!"
sayMe x = "Not between 1 and 5"


-- Pattern matching can help define recursive functions where base cases are represented as 
-- 	individual patterns and recursive cases are also represented as individual patterns
factorial' :: (Integral a) => a -> a
factorial' 0 = 1
factorial' n = n * factorial' (n - 1)


-- Pattern matching can fail if the patterns / cases handled aren't exhausted
-- 	- make a pattern matching construct exhaustive by adding a catchall / default case
charName :: Char -> String
charName 'a' = "Albert"
charName 'b' = "Broseph"
charName 'c' = "Cecil"
charName _ = "Catchall case"


-- Tuples can be used in pattern matching
addVectors :: (Num a) => (a, a) -> (a, a) -> (a, a)
addVectors a b = (fst a + fst b, snd a + snd b)


-- The previous version works, but it can also be done using destructuring to
-- 	bind inputs to local variables
addVectors' :: (Num a) => (a, a) -> (a, a) -> (a, a)
addVectors' (x1, y1) (x2, y2) = (x1 + x2, y1 + y2)


-- You can use pattern matching to destructure input and bind to local variables 
-- 	to the values easier
first :: (a, b, c) -> a
first (x, _, _) = x

second :: (a, b, c) -> b
second (_, y, _) = y

third :: (a, b, c) -> c
third (_, _, z) = z


-- Pattern matching can be used in list comprehensions 
-- let xs = [(1,3), (4,3), (2,4), (5,3), (5,6), (3,1)]
-- [ a+b | (a,b) <- xs ]


-- Lists can be used in pattern matching as well
head' :: [a] -> a
head' [] = error "Can't call head on an empty list!"
head' (x:_) = x


-- Another example of using lists in pattern matching to get more more than just
-- 	the head of the list
tell :: (Show a) => [a] -> String
tell [] = "The list is empty"
tell (x:[]) = "The list has one element: " ++ show x
tell (x:y:[]) = "The list has two elements: " ++ show x ++ " and " ++ show y
tell (x:y:_) = "The list is long.  The first two elements are: " ++ show x ++ " and " ++ show y


-- A reimplementation of the length function using pattern matching, destructuring
-- 	and recursion would look like the following:
length'' :: (Num b) => [a] -> b
length'' [] = 0
length'' (_:xs) = 1 + length'' xs
 

sum' :: (Num a) => [a] -> a
sum' [] = 0
sum' (x:xs) = x + sum' xs


-- In pattern matching, there's something called 'as patterns' (denoted with an '@' symbol) 
-- 	- this allows destructing input and binding to local variables but you an also retain
-- 		a reference to the entirety of the original input
-- 	- in the following example 'all' is a reference to the original input and 'x' and 'xs'
-- 		are destructed and bound local variables
--
-- 	- use 'as patterns' when you want to avoid repeating yourself when matching against a pattern
-- 		and you have to use that whole thing again in the body of the function
capital :: String -> String
capital "" = "Empty string, whoops!"
capital all@(x:xs) = "The first letter of " ++ all ++ " is " ++ [x]
