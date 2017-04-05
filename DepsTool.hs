{-# LANGUAGE RecordWildCards, OverloadedStrings #-}
import Lib
import qualified Data.ByteString.Lazy.Char8 as L
import qualified Data.Map.Strict as M
import Control.Monad
import Data.Function
import Data.List
import System.Environment

mOSTcOMMON = 10000
mAXLEN = 50

mostCommon :: [Sentence] -> M.Map  L.ByteString Int
mostCommon = M.fromList . take mOSTcOMMON . sortBy (compare `on` (negate . snd)) . M.toList . M.fromListWith (+) . map getRaw . concat

getInt m Entry{..} = case (M.lookup entryRaw m,M.lookup entryPos m) of
  (Just x,_) -> x
  (_,Just x) -> x

parseManyFiles :: (L.ByteString -> [a]) -> [String] -> IO [a]
parseManyFiles parser fns = mconcat <$> (forM fns $ \fn -> (do f <- L.readFile fn; return (parser f)))

mkMap :: M.Map L.ByteString Int -> [Sentence] -> M.Map L.ByteString Int
mkMap mc = M.fromList . flip zip [1..] . map fst . M.toList . M.fromListWith (+) . map (getRawOrPos mc) . concat 

readVocab :: FilePath -> IO (M.Map L.ByteString Int)
readVocab fname = (M.fromList . map readDictEntry . filter (not . L.null). L.split '\n') <$> L.readFile fname

sentenceDeps :: L.ByteString -> Dict -> Sentence -> [[L.ByteString]]
sentenceDeps subjLabel vocab sentence = [mkDep verb_index (isPluralVerb verb) subj_index
                              | (subj,subj_index) <- zip sentence [1..],
                                let verb_index = entryParent subj,
                                let verb = sentence !! (verb_index-1),
                                entryLabel subj == subjLabel,
                                subj_index < verb_index,
                                verb_index <= length sentence,
                                isInflected verb]
  where mkDep verb_index isPlural subj_index =
         [bshow n_attractors, bshow (fromEnum isPlural :: Int),
          bshow subj_index, bshow verb_index, showSentence vocab sentence]
         where
          n_attractors = length [() | x<-intervening, isNoun x, isPluralNoun x /= isPlural]
          intervening = take (verb_index - subj_index) $ drop subj_index sentence


main = do
  (cmd:format:fdict:fout:fins) <- getArgs
  let (parser,subjLabel) = case format of
                 "wacky" -> (parseWackySentences,"SBJ")
                 "linzen" -> (parseLinzenSentences,"nsubj")
      parseDeps = parseManyFiles (filter ((<= mAXLEN) . length) . parser) fins
  case cmd of
    "dict" -> do
        mc <- mostCommon <$> parseDeps
        dict <- mkMap mc <$> parseDeps
        L.writeFile fdict $ lineSep $ map showDictEntry $ M.toList dict
    "sentences" -> do
        vocab <- readVocab fdict
        output <- map (showSentence  vocab) <$> parseDeps
        L.writeFile fout (lineSep output)
    "verbs" -> do
        vocab <- readVocab fdict
        output <- (concatMap (sentenceDeps subjLabel vocab)) <$> parseDeps
        L.writeFile fout (lineSep (map tabSep output))

