fizzBuzz :: (Integral a, Show a) => a -> String
fizzBuzz x
  | x `isDivisibleBy` 15 = "FizzBuzz"
  | x `isDivisibleBy` 5  = "Fizz"
  | x `isDivisibleBy` 3  = "Buzz"
  | otherwise            = show x
  where isDivisibleBy divisor dividend = (divisor `mod` dividend) == 0


fizzBuzzes :: (Integral a, Show a) => [a] -> [String]
fizzBuzzes xs = [ fizzBuzz x | x <- xs ]

