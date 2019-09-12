
## Local Development

Run python setup to install the theme:

```
python setup.py install
```

To install the dependencies for this theme, run:

```
pip install -r dependencies/requirements.txt
```


In the root directory install the dependencies of `package.json`:

```
# node version 8.4.0
npm install
```

Now we can run the generated docs in localhost:1919 using :

```
grunt

```

### Grunt options

- 'grunt --project=docs'

This will first look for the path of the numpy/doc in .env file. To make it
work we first need to speficy the path of the numpy source folder.

Example: If you have placed the numpy source code in the same directory of the
scipy-sphinx-theme-v2 then `.env` file will have:

```
{

"DOCS_DIR":"../numpy/doc/source"

}

```

**Note:**

- Sample docs is present on demo-docs folder.
- grunt will automatically refresh the page when we do changes in the docs file.

[TODO]

- Run the docs folder
- Run the devdocs folder
- Run the generatd numpy docs

## Surge deploy

- Every PR will be deployed on surge automatically.
- URL will be pr-<pr_number>-scipy-sphinx-theme-v2.surge.sh
- For example: PR #3 is deployed on https://pr-3-scipy-sphinx-theme-v2.surge.sh
