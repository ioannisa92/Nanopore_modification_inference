
RESDIR="./results/"
GCNINTERPRET="./gcn_interpret_kmers/"
MODELDIR="./saved_models/"

if [ -d "$RESDIR" ]; then
  # Take action if $DIR exists. #
  echo "${RESDIR} already exists"
else
    mkdir $RESDIR
fi

if [ -d "$GCNINTERPRET" ]; then
  # Take action if $DIR exists. #
  echo "${GCNINTERPRET} already exists"
else
    mkdir $GCNINTERPRET
fi

if [ -d "$MODELDIR" ]; then
  # Take action if $DIR exists. #
  echo "${MODELDIR} already exists"
else
    mkdir $MODELDIR
fi

