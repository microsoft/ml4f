#!/bin/sh

set -x
set -e
rm -rf built/gh-pages
mkdir -p built/gh-pages
sed -e 's@src="[^"]*/@src="./@g' < pxt/index.html > built/gh-pages/index.html
cp node_modules/@tensorflow/tfjs/dist/tf.es2017.js built/ml4f.js built/pxtml4f.js built/gh-pages/
touch built/gh-pages/.nojekyll
exit 0
