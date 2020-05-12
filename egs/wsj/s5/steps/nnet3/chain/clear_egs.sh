#!/bin/bash
dir=$1
generate_egs_scp=$2
archives_multiple=$3

echo "$0 $@"
echo "$0: Removing temporary archives, alignments and lattices"
# mkdir -p /tmp/test
# rsync -avP --exclude 'cegs.*.ark' --exclude '*diagnostic.cegs' --exclude '*_uttlist' --exclude 'cmvn_opts' --exclude 'combine.cegs' --exclude 'info' --exclude 'lat_special.scp'  --delete-before /tmp/test/ $dir
# rsync -aqP --exclude "cegs.*.ark" --exclude '*diagnostic.cegs' --exclude '*_uttlist' --exclude 'cmvn_opts' --exclude 'combine.cegs' --exclude "info" --exclude "log" --exclude 'lat_special.scp' --delete /tmp/test/ $dir
# rsync -avz --delete --exclude "cegs.*.ark" --exclude "info" /tmp/test/ test/
cd $dir
for f in $(ls -l . | grep 'cegs_orig' | awk '{ X=NF-1; Y=NF-2; if ($X == "->")  print $Y, $NF; }'); do 
  \rm $f; 
done
echo "$0: Delete cegs_orig link"
# the next statement removes them if we weren't using the soft links to a
# 'storage' directory.
# \rm cegs_orig.*.ark 2>/dev/null
find . -name "cegs_orig.*.ark" -type f -delete
echo "$0: Delete cegs_orig.*.ark"
if ! $generate_egs_scp && [ $archives_multiple -gt 1 ]; then
  # there are some extra soft links that we should delete.
  for f in $dir/cegs.*.*.ark; do \rm $f; done
fi
echo "$0: Delete cegs.*.*.ark"
\rm $dir/ali.{ark,scp} 2>/dev/null
echo "$0: Delete ali.{ark,scp}"
\rm $dir/lat_special.*.{ark,scp} 2>/dev/null
echo "$0: Delete lat_special.*.{ark,scp}"