# HAMR 2015: Deep Music Generation

## Authors

This is a little hack we did at the HAMR@ISMIR 2015 hackday in Málaga!

* Anna Aljanaki, Stefan Balke, Ryan Groves, Eugene Krofto, Eric Nichols

## Hack Goals

* Collect several symbolic song datasets, with melody and possibly chords.
* Represent data in a common vector format appropriate for input to a neural net.
* Develop an LSTM architecture for generation ration melody/chord output.
* Make music!

## Installation

### Python Requirements

```
pip install -r requirements.txt
```

### Datasets

The datasets we use are included. In the following we would like to give credits to the authors:

* Temperley's Rock Corpus
  - http://theory.esm.rochester.edu/rock_corpus/
  - 200 songs
* Essen folk song collection (ESAC): http://www.esac-data.org
* WeimarJazzDatabase: http://jazzomat.hfm-weimar.de

## Documentation

Further documentation can be found on: http://labrosa.ee.columbia.edu/hamr_ismir2015/proceedings/doku.php?id=deepcomposer
