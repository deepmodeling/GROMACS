#!/usr/bin/env python3
#
# This file is part of the GROMACS molecular simulation package.
#
# Copyright (c) 2020, by the GROMACS development team, led by
# Mark Abraham, David van der Spoel, Berk Hess, and Erik Lindahl,
# and including many others, as listed in the AUTHORS file in the
# top-level source directory and at http://www.gromacs.org.
#
# GROMACS is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License
# as published by the Free Software Foundation; either version 2.1
# of the License, or (at your option) any later version.
#
# GROMACS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with GROMACS; if not, see
# http://www.gnu.org/licenses, or write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
#
# If you want to redistribute modifications to GROMACS, please
# consider that scientific software is very special. Version
# control is crucial - bugs must be traceable. We will be happy to
# consider code for inclusion in the official distribution, but
# derived work must not be called official GROMACS. Details are found
# in the README & COPYING files - if they are missing, get the
# official version at http://www.gromacs.org.
#
# To help us fund GROMACS development, we humbly ask that you cite
# the research papers on the package. Check out http://www.gromacs.org.


import argparse
import enum
import json
import logging
import sys
import time
import urllib.parse
import urllib.request


@enum.unique
class PipelineStatus(enum.Enum):
    PENDING = 0
    SUCCESS = 1
    FAILURE = 2


SLEEP = 15  # seconds


def get_pipelines_for_commit(project, sha, token=None):
    project_encoded = urllib.parse.quote_plus(project)
    url = 'https://gitlab.com/api/v4/projects/{project}/pipelines?sha={sha}'.format(
        project=project_encoded, sha=sha
    )
    headers = dict()
    if token is not None:
        headers['Authorization'] = 'Bearer {}'.format(token)
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request) as fp:
            return json.load(fp)
    except urllib.error.HTTPError:
        logging.exception('Can not get {} with token {}'.format(url, str(token)))
        raise


def pipeline_is_not_mr(pipeline):
    ref = pipeline['ref']
    # MR pipelines have ref="refs/merge-requests/{mr_number}/head"
    # push pipelines have ref="{branch_name}"
    # after-merge and nightly pipelines have ref="{branch_name}"
    return not ref.startswith('refs/merge-requests/')


def get_pipeline_status(pipeline):
    status = pipeline['status']
    if status in ('created', 'waiting_for_resource', 'preparing', 'pending', 'running', 'scheduled', 'manual'):
        return PipelineStatus.PENDING
    elif status in ('success',):
        return PipelineStatus.SUCCESS
    elif status in ('failed', 'canceled', 'skipped'):
        return PipelineStatus.FAILURE
    else:
        raise RuntimeError('Unexpected pipeline status: {}'.format(status))


parser = argparse.ArgumentParser()
parser.add_argument('--project', type=str, required=False,
                    default='gromacs/gromacs',
                    help='Name of the GitLab project')
parser.add_argument('--timeout', type=int, required=False,
                    default=1800,
                    help='Timeout in seconds to wait for pipeline to finish')
parser.add_argument('--token', type=str, required=False,
                    help='GitLab API token')
parser.add_argument('--sha', type=str, required=True,
                    help='SHA of commit to check')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    end_time = time.monotonic() + args.timeout

    while time.monotonic() < end_time:
        pipelines = get_pipelines_for_commit(args.project, args.sha, args.token)
        try:
            statuses = {
                pipeline['id']: get_pipeline_status(pipeline)
                for pipeline in pipelines
                if pipeline_is_not_mr(pipeline)
            }
            if any(s == PipelineStatus.FAILURE for s in statuses.values()):
                raise RuntimeError('One of the pipelines failed')
            if all(s == PipelineStatus.SUCCESS for s in statuses.values()):
                logging.info('All pipelines succeeded')
                sys.exit(0)
        except Exception as exc:
            logging.info('Data from GitLab:\n' + json.dumps(pipelines, indent=2))
            logging.exception(exc)
            raise
        time.sleep(SLEEP)
    raise TimeoutError('Pipelines did non complete in time')

