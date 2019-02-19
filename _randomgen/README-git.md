These are the bash commands used to get the bashtage/randomgen repo into numpy/numpy

```bash
# from a directory just above a numpy git checkout
git clone https://github.com/bashtage/randomgen.git
cd randomgen
# rewrite the checkout, pushing the content into '_randomgen'
git filter-branch --index-filter '
    git ls-files -s |
    sed "s-\t-\t_randomgen/-" |
    GIT_INDEX_FILE=$GIT_INDEX_FILE.new git update-index --index-info &&
    mv $GIT_INDEX_FILE.new $GIT_INDEX_FILE
' HEAD
# write this file, commit it
git add _randomgen/README-git.md
git commit -m"Add README-git.md"
git checkout -b randomgen
cd ../numpy
git checkout -b randomgen
git remote add randomgen ../randomgen
git fetch randomgen randomgen
git merge --allow-unrelated-histories randomgen/randomgen
git remote remove randomgen
# Now all the randomgen commits are on the randomgen branch in numpy,
# and there is a subdirectory _randomgen with the content
```
