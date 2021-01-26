#!/bin/bash
set -euo pipefail

# This script performs the automatic upload of release artefacts to the
# GROMACS FTP and manual servers once a release has been completed and all artefacts
# have been successfully generated.

# The script will only run under the following conditions (in addition to the general
# GitLab pipeline rules that govern what runs for release builds):
#
# * The branch needs to be a release branch (or master for beta releases)
# * The SSH keys for uploading need to be available in the GitLab variable "SSH_PRIVATE_KEY"
#
# If the variable GROMACS_RELEASE is not == true, then any upload will go to a dummy location
# on the FTP and manual servers, to check that things work. This is needed for testing this script.
# When not on one of the recognized release branches, we also set the script to use test locations.

# By default, assume we are doing the full upload to the default locations
# As both the manual and ftpbot can't place files anywhere except their base
# directories (/srv/ftp for the ftpbot and /var/www/manual for the manualbot),
# we set the upload path relative to this.
UPLOAD_LOCATION_FTP=./ # is equivalent to /srv/ftp
UPLOAD_LOCATION_WWW=./ # is equivalent to /var/www/manual
UPLOAD_REMOTE_FTP=ftpbot@ftp.gromacs.org
UPLOAD_REMOTE_WWW=manualbot@manual.gromacs.org

if [[ "${CI_COMMIT_REF_NAME}" != "master" && "${CI_COMMIT_REF_NAME}" != "release-2021" && "${CI_COMMIT_REF_NAME}" != "release-2020" ]] ; then
    echo "Not running for any recognized branch, not uploading to the real locations"
    UPLOAD_LOCATION_FTP="${UPLOAD_LOCATION_FTP}/.ci-test"
    UPLOAD_LOCATION_WWW="${UPLOAD_LOCATION_WWW}/.ci-test"
elif [[ "${GROMACS_RELEASE}" != "true" ]] ; then
    echo "Not running a true release build, not uploading to the real locations"
    UPLOAD_LOCATION_FTP="${UPLOAD_LOCATION_FTP}/.ci-test"
    UPLOAD_LOCATION_WWW="${UPLOAD_LOCATION_WWW}/.ci-test"
fi

echo "Running upload to FTP (${UPLOAD_LOCATION_FTP}), WWW (${UPLOAD_LOCATION_WWW}) servers"
echo "We are uploading files for this version: ${VERSION}"

# Get files for uploading the manual front page
MANUAL_PAGE_REPO=manual-front-page
if [[ -d "${MANUAL_PAGE_REPO}" ]]; then
    rm -rf "${MANUAL_PAGE_REPO}"
fi
git clone --depth=1 git@gitlab.com:gromacs/deployment/manual-front-page.git


SPHINX=$(which sphinx-build)
if [[ -z "${SPHINX}" ]] ; then
    echo "Error Can't do things without having sphinx available"
    exit 1
fi

originalpwd="${PWD}"
(
    
    upload="rsync -rlptvP --chmod=u+rwX,g+rwX,o+rX"
    deploymentlocation="${UPLOAD_REMOTE_WWW}:${UPLOAD_LOCATION_WWW}"
    website_loc="${BUILD_DIR}/docs/html/"
    if [[ ! -d "${website_loc}" ]] ; then
        echo "Error Can't find the webpage files"
        exit 1
    fi
    ${upload} "${website_loc}"/* "${website_loc}"/.[a-z]* "${deploymentlocation}"/"${VERSION}"/
    echo "done upload"
    cp "${website_loc}/manual-${VERSION}.pdf" "${originalpwd}"
)


(
    upload="rsync -rlptvP --chmod=u+rw,g+rw,o+r"
    destination="${UPLOAD_REMOTE_FTP}:${UPLOAD_LOCATION_FTP}"
    regressiontests_tarball="regressiontests-${VERSION}.tar.gz"
    source_tarball="gromacs-${VERSION}.tar.gz"
    md5sum "${source_tarball}"
    md5sum "${regressiontests_tarball}"
    ${upload} "${source_tarball}" "${destination}/gromacs/"
    ${upload} "manual-${VERSION}.pdf" "${destination}/manual/"
    ${upload} "${regressiontests_tarball}" "${destination}/regressiontests/"
)

(
    cd "${MANUAL_PAGE_REPO}"
    upload="rsync -rlptvP --chmod=u+rwX,g+rwX,o+rX"
    deploymentlocation="${UPLOAD_REMOTE_WWW}:${UPLOAD_LOCATION_WWW}"
    make html
    # we always fail save
    ${upload} _build/html/ "${deploymentlocation}/" --exclude _sources --exclude .buildinfo --exclude objects.inv
)

