{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TupleSections, OverloadedStrings #-}
module Lib where
import qualified Data.ByteString.Lazy.Char8 as L
import qualified Data.Map.Strict as M
import Control.Monad
import Data.Monoid
import Data.Char
import Data.List.Split

type Dict = M.Map L.ByteString Int

data Entry = Entry {entryParent :: !Int,
                    entryRaw :: !L.ByteString,
                    entryPos :: !L.ByteString,
                    entryLabel :: !L.ByteString}
             deriving Show
type Sentence = [Entry]

entryWord :: Dict -> Entry ->  Int
entryWord m Entry{..} = case (M.lookup entryRaw m,M.lookup entryPos m) of
  (Just x,_) -> x
  (_,Just x) -> x
  x -> error ("Input word not found in vocabulary: " ++ show (entryRaw, entryPos))

showSentence :: Dict -> Sentence -> L.ByteString
showSentence intMap = L.intercalate (L.pack " ") . map (L.pack . show . entryWord intMap)

getRaw Entry{..} = (entryRaw,1::Int)
getRawOrPos m Entry{..} = (,1::Int) $ case M.lookup entryRaw m of
  Just _ -> entryRaw
  Nothing -> entryPos

showDictEntry :: (L.ByteString,Int) -> L.ByteString
showDictEntry (wd,num) = tabSep [wd,bshow num]

readDictEntry :: L.ByteString -> (L.ByteString,Int)
readDictEntry e = (wd,read (L.unpack num))
  where (wd,num) = case L.split '\t' e of
          [x,y] -> (x,y)
          _ -> error $ "Could not parse dict entry: " ++ show e

readManyUTFFiles :: [String] -> IO L.ByteString
readManyUTFFiles fns = do
  mconcat <$> (forM fns L.readFile)

isPluralVerb Entry{..} = entryPos `elem` ["VBP", "VVP", "VHP"]
isNoun  Entry{..} = entryPos `elem` ["NN", "NNS"]
isPluralNoun  Entry{..} = entryPos `elem` ["NNS"]

isInflected Entry{..} = entryPos `elem` ["VBP", "VVP", "VHP", "VBZ", "VVZ", "VHZ"]

tabSep = L.intercalate "\t"
lineSep = L.concat . map (<> L.pack "\n")

splitLines = L.split '\n'

bshow = L.pack . show

bread :: Read a => L.ByteString -> a
bread = read . L.unpack

parseLinzenSentences :: L.ByteString -> [Sentence]
parseLinzenSentences = map (map (cleanEntry . L.split '\t')) .
                  filter (not . null) . splitWhen (L.null) . splitLines
  where cleanEntry :: [L.ByteString] -> Entry
        cleanEntry [_index,raw,_,pos,pos',_,parent,label,_,_] = 
          Entry {entryParent = bread parent
                ,entryRaw = L.map toLower raw
                ,entryPos = pos
                ,entryLabel = label}
        cleanEntry x = error $ "CleanEntry: " ++ show x

parseWackySentences :: L.ByteString -> [Sentence]
parseWackySentences = map (map cleanEntry . filter (not . null) . map (L.split '\t')) .
            filter (not . null) . splitWhen (L.pack "<" `L.isPrefixOf`) .  splitLines
  where cleanEntry :: [L.ByteString] -> Entry
        cleanEntry [raw, _lemma, pos, _index, parent, label] =
          Entry {entryParent = bread parent
                ,entryRaw = L.map toLower raw
                ,entryPos = pos
                ,entryLabel = label}
        cleanEntry x = error $ "CleanEntry: " ++ show x

