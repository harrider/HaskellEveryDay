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


-- Guard conditions can be used with pattern matching to help test the inputs for a matching pattern
-- 	- "otherwise" is the idiomatic catchall when using guard conditions in Haskell
--
-- 	- if no "otherwise" guard condition is supplied to some pattern, then the logic will fall
-- 		through to the function's next pattern in the list of patterns to match against
bmiTell :: (RealFloat a) => a -> String
bmiTell bmi
  | bmi <= 18.5 = "You're underweight, you emo, you!"
  | bmi <= 25.0 = "You're supposedly normal.  Pffft, I bet you're an ugly!"
  | bmi <= 30.0 = "You're fat!  Lose some weight, fatty!"
  | otherwise = "You're a what, congrats."


-- Instead of relying on the bmi being passed into the function above, we can calculate it ourselves
-- 	inside of the guard conditions
bmiTell' :: (RealFloat a) => a -> a -> String
bmiTell' weight height
  | weight / height ^ 2 <= 18.5 = "You're underweight, you emo, you!"
  | weight / height ^ 2 <= 25.0 = "You're supposedly normal.  Pffft, I bet you're an ugly!"
  | weight / height ^ 2 <= 30.0 = "You're fat!  Lose some weight, fatty!"
  | otherwise = "You're a what, congrats."


-- Another example of using guard conditions with pattern matching
max' :: (Ord a) => a -> a -> a
max' a b
  | a > b     = a
  | otherwise = b


-- Guard conditions can be written inline as well, but don't do this as it's hard to read
max'' :: (Ord a) => a -> a -> a
max'' a b | a > b = a | otherwise = b


-- Another example of using guard conditions with pattern mathcing
myCompare :: (Ord a) => a -> a -> Ordering
a `myCompare` b
  | a > b     = GT
  | a < b     = LT
  | otherwise = EQ


-- In the bmiTell' function above, we defined a function 3 times... we can use 
-- 	the 'where' operator to define a local function and use in all of the
-- 	guard conditions to adhere to the DRY principle
-- 
bmiTell'' :: (RealFloat a) => a -> a -> String
bmiTell'' weight height
  | bmi <= 18.5 = "You're underweight, you emo, you!"
  | bmi <= 25.0 = "You're supposedly normal.  Pffft, I bet you're an ugly!"
  | bmi <= 30.0 = "You're fat!  Lose some weight, fatty!"
  | otherwise = "You're a what, congrats."
  where bmi = weight / height ^ 2


-- We can define functions and variables after the 'where' keyword that an be
-- 	used inside of the guard conditions
--
-- 	- These local functions and variables are only in scope for the function 
-- 		where the 'where' keyword was used and only for the pattern it
-- 		was defined under
--
-- 	- NOTE: "Where bindings aren't shared across function bodies of different patterns.
-- 		If you want several patterns of one function to access some shared name,
-- 		you have to define it globall"
bmiTell''' :: (RealFloat a) => a -> a -> String
bmiTell''' weight height
  | bmi <= skinny = "You're underweight, you emo, you!"
  | bmi <= normal = "You're supposedly normal.  Pffft, I bet you're an ugly!"
  | bmi <= fat = "You're fat!  Lose some weight, fatty!"
  | otherwise = "You're a what, congrats."
  where bmi = weight / height ^ 2
        skinny = 18.5
        normal = 25.0
        fat = 30.0


-- you can use where bindings to pattern match as well
-- 	- a triple is destructured and bound to the variables
bmiTell'''' :: (RealFloat a) => a -> a -> String
bmiTell'''' weight height
  | bmi <= skinny = "You're underweight, you emo, you!"
  | bmi <= normal = "You're supposedly normal.  Pffft, I bet you're an ugly!"
  | bmi <= fat = "You're fat!  Lose some weight, fatty!"
  | otherwise = "You're a what, congrats."
  where bmi = weight / height ^ 2
        (skinny, normal, fat) = (18.5, 25.0, 30.0)


-- In the following function, we use the where clause to pattern match and 
-- 	destructure the input strings into local variable bindings
initials :: String -> String -> String
initials firstname lastname = [f] ++ ". " ++ [l] ++ "."
  where (f:_) = firstname
        (l:_) = lastname


-- We use the 'where' keyword to define a local function and use that function in
-- 	the list comprehension to calculate a list of bmis
--
-- We also use pattern matching in the list comprehension to destructure the list of
-- 	pairs into local variable bindings (w, h) that are used as arguments for our
-- 	local function
calcBmis :: (RealFloat a) => [(a,a)] -> [a]
calcBmis xs = [ bmi w h | (w, h) <- xs ]
  where bmi weight height = weight / height ^ 2


-- 'let' bindings allow you to bind variables anywhere and they are expressions
-- 	- they are very local and the bindings do not span across guard conditions
--
-- 	- 'let' bindings can be used for pattern matching as well
--
-- The form for let bindings is:
-- 	let <bindings> in <expression>
--
-- 	- The names / variables that you define in the 'let' part are accessible
-- 		in the expression part that comes afterwards
--
-- 	- NOTE: you have to add an '=' sign after the function name and arguments
-- 		for 'let' bindings unlike with 'where' bindings
--
-- 'let' bindings are expressions themselves whereas 'where' bindings are just 
-- 	syntactic constructs
cylinder :: (RealFloat a) => a -> a -> a
cylinder r h =
  let sideArea = 2 * pi * r * h
      topArea = pi * r ^ 2
  in  sideArea + 2 * topArea



-- 'let' bindings can be used in list comprehensions to introduce bindings and functions
-- 	in a local scope to the list comprehension
calculateSquares :: (Num a) => [a] -> [a]
calculateSquares xs = [ let square x = x * x in square a | a <- xs ]


-- In order to inline bind multiple names / variables in a 'let' binding, you add 
-- 	a ';' (semi-colon) between bindings
--
-- 	- NOTE: you only use the 'let' keyword ONCE and separate the bindings without additional
-- 		'let' keywords with a semi-colon / ;
--
-- 		(e.g) let x = 1; y = 2; z = 3; in addThree a b c = x + y + z
multiplyPairs :: (Num a) => [(a,a)] -> [a]
multiplyPairs xs = [ let w = 1; h = 1; op l r = l * r * w * h; in op f s | (f, s) <- xs ]


-- To make the list comprehension easier to read when using 'let' bindings, you can add the
-- 	'let' binding to the end of the list comprehension
calculateSquares' :: (Num a) => [a] -> [a]
calculateSquares' xs = [ square a | a <- xs, let square x = x * x ]


-- Another example where we rewrite the bmi calculator using 'let' bindings in a list comprehension
calcBmis' :: (RealFloat a) => [(a,a)] -> [a]
calcBmis' xs = [ bmi | (w,h) <- xs, let bmi = w / h ^ 2 ]


-- NOTE: The 'in' part of a 'let' binding can be omitted in a list comprehension because the 
-- 	visibility of the names is already predefined there.
--
-- 	- However, the 'in' part of the 'let' binding could be used in a predicate and the names
-- 		used in the binding would only be visible to that predicate



-- Case Expressions are essentially 'switch' expressions in Haskell
-- 	- Pattern matching in function definitions is syntactic sugar over case expressions
--
-- Case Expression form looks like:
-- 	case <expression> of <pattern> -> <result>
-- 			     <pattern> -> <result>
-- 			     <pattern> -> <result>
-- 			     ...

-- NOTE: Why would you use case expressions instead of pattern matching if pattern matching is
-- 	syntactic sugar over case expressions?
--
-- 	- pattern matching on function parameters can only be done when defining functions
--
-- 	- case expressions can be used pretty much anywhere
-- 		- they are used for pattern matching against something in the middle of an expression
describeList :: [a] -> String
describeList xs = "The list is " ++ case xs of [] -> "empty."
                                               [x] -> "a single list."
                                               xs -> "a longer list."

-- Because pattern matching in function definitions is syntactic sugar for case expressions,
-- 	we could also have defined the above like this:
describeList' :: [a] -> String
describeList' xs = "The list is " ++ what xs
  where what [] = "empty."
        what [x] = "a single list."
        what xs = "a longer list."


-- Recursion is used in Haskell because in Haskell you declare what something is rather than how to get it
-- 	- Pattern matching, guard conditions and recursion all go together well
maximum' :: (Ord a) => [a] -> a
maximum' [] = error "maximum of empty list"
maximum' [x] = x
maximum' (x:xs)
  | x > maxTail = x
  | otherwise = maxTail
  where maxTail = maximum' xs


-- A clearer way of writing the above would be to use the max function in the recursive pattern
maximum'' :: (Ord a) => [a] -> a
maximum'' [] = error "maximum of empty list"
maximum'' [x] = x
maximum'' (x:xs) = max x (maximum' xs)


-- Using recursion to build a list of values recursively
replicate' :: (Num i, Ord i) => i -> a -> [a]
replicate' n x
  | n <= 0    = []
  | otherwise = x:replicate' (n-1) x


-- Using recursion to take a certain number of elements form a list
take' :: (Num i, Ord i) => i -> [a] -> [a]
take' n _
  | n <= 0     = []
take' _ []     = []
take' n (x:xs) = x:take' (n-1) xs


-- Using recurion to reverse a list of numbers
reverse' :: [a] -> [a]
reverse' [] = []
reverse' (x:xs) = reverse' xs ++ [x]


-- Using recurion to create an infinite list
repeat' :: a -> [a]
repeat' x = x:repeat' x


-- Using recursion to create a list of pairs from 2 separate lists
zip' :: [a] -> [b] -> [(a,b)]
zip' _ [] = []
zip' [] _ = []
zip' (x:xs) (y:ys) = (x,y):zip' xs ys


-- Using recurion to see if an element exists within a list
elem' :: (Eq a) => a -> [a] -> Bool
elem' a [] = False
elem' a (x:xs)
  | a == x     = True
  | otherwise  = a `elem'` xs

 
-- Implementing Quicksort in Haskell
quicksort :: (Ord a) => [a] -> [a]
quicksort [] = []
quicksort (x:xs) =
  let smallerSorted = quicksort [ a | a <- xs, a <= x ]
      biggerSorted = quicksort [ a | a <- xs, a > x ]
  in smallerSorted ++ [x] ++ biggerSorted


-- Curried function - Every function in Haskell takes only one parameter and either returns
-- 	a value or another function.
--
-- 	- A function that does not receive all of the parameters listed in its type signature
-- 		becomes a paritally applied function that accepts as many parameters that 
-- 		were not applied previously.
--
-- 	- The ' ' (space) between function parameters is called "function application" and it
-- 		has the highest precedence in Haskell 
multThree :: (Num a) => a -> a -> a -> a
multThree x y z = x * y * z 

-- -- Examples of partial function application in Haskell
-- let multTwoWithNine = multThree 9
-- multTwoWithNine 2 3  // output = 54

-- let multWithEighteen = multTwoWithNine 2
-- multWithEighteen 10  // output = 180


-- Defining a function to compare numbers to 100
compareWithHundred :: (Num a, Ord a) => a -> Ordering
compareWithHundred x = compare 100 x


-- The above can be re-written using a partially applied function
compareWithHundred' :: (Num a, Ord a) => a -> Ordering
compareWithHundred' = compare 100

-- -- Looking at the type signatures of the 2 functions above, we can see that they're the same
-- ghci> :t compareWithHundred
-- compareWithHundred :: (Num a, Ord a) => a -> Ordering
-- ghci> :t compareWithHundred'
-- compareWithHundred' :: (Num a, Ord a) => a -> Ordering



-- Infix function can also be partially applied by using "sections"
-- 	- To section an infix function, surround it with parenthesis / () and only supply a
-- 		parameter on one side.
-- 		- That creates a function that only takes one parameter and then applies it ot the 
-- 			side that's missing an operand
divideByTen :: (Floating a) => a -> a
divideByTen = (/10)


-- calling the partially applied infix function would look like:
-- ghci> divideByTen 200   // this would be equivalent to 200 / 10

-- the above is also equivalent to:
-- ghci> (/10) 200	   // which again would be equivalent to 200 / 10


-- The following function would check to see if a character supplied to it is an uppercase letter
isUpperAlphanum :: Char -> Bool
isUpperAlphanum = (`elem` ['A'..'Z'])


-- The only special thing about "sections" is when trying to create a section for partially applying
-- 	subtraction. 
--
-- 	- (-4) does not create a partially applied subtraction function.  This would actually be 
-- 		negative 4
--
-- 	- In order to create a partially applied subtraction function, partially apply the
-- 		"subtract" function:
--
-- 		- (subtract 4)



-- Functions can take functions as parameters and also return functions.  The following function
-- 	tkaes a function as an input and applies it twice to something
applyTwice :: (a -> a) -> a -> a
applyTwice f x = f (f x)


-- -- Running the above function with various inputs gives the following results:
-- ghci> applyTwice (+3) 10    		// output = 16  ... (+3) is a partially applied function
-- ghci> applyTwice (++ " HAHA") "HEY"  // output = "HEY HAHA HAHA"
-- ghci> applyTwice ("HAHA " ++) "HEY"	// output = "HAHA HAHA HEY"
-- ghci> applyTwice (multThree 2 2) 9	// output = 144
-- ghci> applyTwice (3:) [1]		// output = [3,3,1]



-- Reimplementing a standard library function (zipWith) using higher order programming
zipWith' :: (a -> b -> c) -> [a] -> [b] -> [c]
zipWith' _ [] _ = []
zipWith' _ _ [] = []
zipWith' f (x:xs) (y:ys) = f x y : zipWith' f xs ys


-- -- Trying out zipWith' produces the following results:
-- ghci> zipWith' (+) [4,2,5,6] [2,6,2,3]		// output = [6,8,7,8]
-- ghci> zipWith' max [6,3,2,1] [7,3,1,5]		// output = [7,3,1,5]
-- ghci> zipWith' (++) ["foo ", "bar ", "baz "] ["fighters", "hoppers", "aldrin"]  // output = ["foo fighters", "bar hoppers", "baz aldrin"]
-- ghci> zipWith' (*) (replicate 5 2) [1..]	// output = [2, 4, 6, 8, 10]
-- ghci> zipWith' (zipWith' (*)) [[1,2,3],[3,5,6],[2,3,4]] [[3,2,2],[3,4,5],[5,4,3]]  // output = [[3,4,6],[9,20,30],[10,12,12]]



-- Reimplementing another function from the standard library
flip' :: (a -> b -> c) -> (b -> a -> c)
flip' f = g
  where g x y = f y x


-- A simpler way to define the above function would be:
flip'' :: (a -> b -> c) -> b -> a -> c
flip'' f y x = f x y


-- -- Using the flip' function produces the following results:
-- ghci> flip' zip [1,2,3,4,5] "hello"	// output = [('h',1),('e',2),('l',3),('l',4),('o',5)]
-- ghci> zipWith (flip' div) [2,2..] [10,8,6,4,2]	// output = [5,4,3,2,1]



