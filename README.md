# DNNSyntax
Experimenting with prediction of verb number using DNNs


# Preparation of data

With access to the Wackypedia corpus, you can prepare the data like so:

```
ghc --make DepsTool

./DepsTool dict wacky <vocabulary file> - <corpus files>
./DepsTool sentences wacky <vocabulary file> <sentences file> <corpus files>
./DepsTool deps wacky <vocabulary file> <subj-verb deps file> <corpus files>
```


# LM experiments

Edit `lm.py` to set the input parameters as you desire. Then:

```
python lm.py create train <number of epochs> inflection <subj-verb deps file> prediction <subj-verb deps file>
```

# Supervised number prediction

```
python inflect.py create train <number of epochs> test
```
