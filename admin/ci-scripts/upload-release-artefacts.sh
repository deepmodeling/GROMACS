#!/bin/bash
set -euo pipefail

# This script performs the automatic upload of release artefacts to the
# GROMACS FTP and WWW servers once a release has been completed and all artefacts
# have been successfully generated.

# The script will only run under the following conditions (in addition to the general
# GitLab pipeline rules that govern what runs for release builds):
#
# * The branch needs to be a release branch (or master for beta releases)
# * The SSH keys for uploading need to be available in the GitLab variable "SSH_PRIVATE_KEY"
#
# If the variable GROMACS_RELEASE is not == true, then any upload will go to a dummy location
# on the WWW and FTP servers, to check that things work. This is needed for testing this script.
# When not on one of the recognized release branches, we also set the script to use test locations.

# By default, assume we are doing the full upload
UPLOAD_LOCATION_FTP=/srv/ftp
UPLOAD_LOCATION_WWW=/var/www/manual

if [[ "$CI_COMMIT_REF_NAME" != "master" && "$CI_COMMIT_REF_NAME" != "release-2021" && "$CI_COMMIT_REF_NAME" != "release-2020" ]] ; then
    echo "Not running for any recognized branch, not uploading to the real locations"
    UPLOAD_LOCATION_FTP=$UPLOAD_LOCATION_FTP/test
    UPLOAD_LOCATION_WWW=$UPLOAD_LOCATION_WWW/test
else if [[ "$GROMACS_RELEASE" != "true" ]] ; then
    echo "Not running a true release build, not uploading to the real locations"
    UPLOAD_LOCATION_FTP=$UPLOAD_LOCATION_FTP/test
    UPLOAD_LOCATION_WWW=$UPLOAD_LOCATION_WWW/test
fi

echo "Running to upload to FTP ($UPLOAD_LOCATION_FTP), WWW ($UPLOAD_LOCATION_WWW)"
echo "We are uploading files for this version: $VERSION"

# Get files for uploading the manual front page
MANUAL_PAGE_REPO=manual-front-page
if [[ ! -d $MANUAL_PAGE_REPO ]] ; then
    mkdir $MANUAL_PAGE_REPO
    cd $MANUAL_PAGE_REPO
    git init
    cd ..
fi

cd $MANUAL_PAGE_REPO
git fetch git@gitlab.com:gromacs/deployment/manual-front-page.git master --depth=1
git checkout -qf FETCH_HEAD
git clean -ffdxq
git gc
cd ..

SPHINX=`which sphinx-build`
if [[ -z $SPHINX ]] ; then
    echo "Can't do things without having sphinx available"
    exit 1
fi

originalpwd=`pwd`
(
    
    upload="rsync -avP --chmod=u+rwX,g+rwX,o+rX"
    deploymentlocation="pbauer@www.gromacs.org:$UPLOAD_LOCATION_WWW"
    website_loc=$BUILD_DIR/docs/html/
    if [[ ! -d $website_loc ]] ; then
        echo "Can't find the webpage files"
        exit 1
    fi
    #$upload $website_loc/* $website_loc/.[a-z]* $deploymentlocation/${VERSION}/
    echo "done upload"
    cp $website_loc/manual-${VERSION}.pdf $originalpwd
)


(
    upload="rsync -avP --chmod=u+rw,g+rw,o+r"
    destination="pbauer@ftp.gromacs.org:$UPLOAD_LOCATION_FTP"
    regressiontests_tarball="regressiontests-${VERSION}.tar.gz"
    source_tarball="gromacs-${VERSION}.tar.gz"
    md5sum $source_tarball
    md5sum $regressiontests_tarball
    #$upload $source_tarball $destination/gromacs/
    #$upload manual-${VERSION}.pdf $destination/manual/
    #$upload $regressiontests_tarball $destination/regressiontests/
)

(
    cd $MANUAL_PAGE_REPO
    upload="rsync -avP --chmod=u+rwX,g+rwX,o+rX"
    deploymentlocation="pbauer@www.gromacs.org:$UPLOAD_LOCATION_WWW"
    make html
    #$upload -av _build/html/ $deploymentlocation/ --exclude _sources --exclude .buildinfo --exclude objects.inv
)

