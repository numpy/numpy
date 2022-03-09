set_travis_vars() {
    # Set env vars
    echo "TRAVIS_EVENT_TYPE is $TRAVIS_EVENT_TYPE"
    echo "TRAVIS_TAG is $TRAVIS_TAG"
    if [[ "$TRAVIS_EVENT_TYPE" == "push" && "$TRAVIS_TAG" == refs/tags/v* ]]; then
      IS_PUSH="true"
    else
      IS_PUSH="false"
    fi
    if [[ "$TRAVIS_EVENT_TYPE" == "cron" ]]; then
      IS_SCHEDULE_DISPATCH="true"
    elif [[ "$TRAVIS_EVENT_TYPE" == "api" ]]; then
      # Manual CI run, so upload
      IS_SCHEDULE_DISPATCH="true"
    else
      IS_SCHEDULE_DISPATCH="false"
    fi
}
set_upload_vars() {
    echo "IS_PUSH is $IS_PUSH"
    echo "IS_SCHEDULE_DISPATCH is $IS_SCHEDULE_DISPATCH"
    if [[ "$IS_PUSH" == "true" ]]; then
        echo push and tag event
        export ANACONDA_ORG="multibuild-wheels-staging"
        export TOKEN="$NUMPY_STAGING_UPLOAD_TOKEN"
        export ANACONDA_UPLOAD="true"
    elif [[ "$IS_SCHEDULE_DISPATCH" == "true" ]]; then
        echo scheduled or dispatched event
        export ANACONDA_ORG="scipy-wheels-nightly"
        export TOKEN="$NUMPY_NIGHTLY_UPLOAD_TOKEN"
        export ANACONDA_UPLOAD="true"
    else
        echo non-dispatch event
        export ANACONDA_UPLOAD="false"
    fi
}
upload_wheels() {
    echo ${PWD}
    if [[ ${ANACONDA_UPLOAD} == true ]]; then
        if [ -z ${TOKEN} ]; then
            echo no token set, not uploading
        else
            python -m pip install git+https://github.com/Anaconda-Server/anaconda-client
            ls ./wheelhouse/*.whl
            anaconda -t ${TOKEN} upload --no-progress --skip -u ${ANACONDA_ORG} ./wheelhouse/*.whl
            echo "PyPI-style index: https://pypi.anaconda.org/$ANACONDA_ORG/simple"
        fi
    fi
}
