#! /usr/bin/env python3

import os
import sys
from subprocess import check_output #nosec

# add .. to PYTHONPATH
path = os.path.realpath(__file__)
basedir = os.path.dirname(os.path.dirname(path))
sys.path.append(basedir)

# add ../blind and ../util to PATH
os.environ['PATH'] += ':' + os.path.join(basedir, 'blind')
os.environ['PATH'] += ':' + os.path.join(basedir, 'util')

os.environ['DJANGO_SETTINGS_MODULE'] = 'astrometry.net.settings'

import django
django.setup()

import tempfile
import traceback
from urllib.parse import urlparse
import urllib.request, urllib.parse, urllib.error
import shutil
import multiprocessing
import time
import re
import tarfile
import gzip
import zipfile
import math

from astrometry.util import image2pnm
from astrometry.util.filetype import filetype_short
from astrometry.util.run_command import run_command

from astrometry.util.util import Tan
from astrometry.util import util as anutil
from astrometry.util.fits import *

import settings
settings.LOGGING['loggers'][''] = {
    'handlers': ['console'],
    'level': 'INFO',
    'propagate': True,
}
from astrometry.net.models import *
from log import *

from django.db.models import Count
from django.db import DatabaseError
from django.db.models import Q

from logging.config import dictConfig
dictConfig(settings.LOGGING)

import logging
logging.basicConfig(format='%(message)s', level=logging.DEBUG)

def is_tarball(fn):
    logmsg('is_tarball: %s' % fn)
    types = filetype_short(fn)
    logmsg('filetypes:', types)
    for t in types:
        if t.startswith('POSIX tar archive'):
            return True
    return False

def get_tarball_files(fn):
    # create temp dir to extract tarfile.
    tempdir = tempfile.mkdtemp()
    cmd = 'tar xvf %s -C %s' % (fn, tempdir)
    #userlog('Extracting tarball...')
    (rtn, out, err) = run_command(cmd)
    if rtn:
        #userlog('Failed to un-tar file:\n' + err)
        #bailout(submission, 'failed to extract tar file')
        print('failed to extract tar file')
        return None
    fns = out.strip('\n').split('\n')

    validpaths = []
    for fn in fns:
        path = os.path.join(tempdir, fn)
        logmsg('Path "%s"' % path)
        if not os.path.exists(path):
            logmsg('Path "%s" does not exist.' % path)
            continue
        if os.path.islink(path):
            logmsg('Path "%s" is a symlink.' % path)
            continue
        if os.path.isfile(path):
            validpaths.append(path)
        else:
            logmsg('Path "%s" is not a file.' % path)

    if len(validpaths) == 0:
        #userlog('Tar file contains no regular files.')
        #bailout(submission, "tar file contains no regular files.")
        #return -1
        logmsg('No real files in tar file')
        return None

    logmsg('Got %i paths.' % len(validpaths))
    return validpaths

def run_pnmfile(fn):
    """ run the pnmfile command to get image dimenensions"""

    # bandit warns about security implications. We provide our own filename
    # and the sysadmin has to make sure the correct pnmfile is found in the PATH
    # TODO: document this for sysadmins
    out = check_output(['pnmfile', fn]).decode().strip() #nosec
    logmsg('pnmfile output: ' + out)
    pat = re.compile(r'P(?P<pnmtype>[BGP])M .*, (?P<width>\d*) by (?P<height>\d*)( *maxval (?P<maxval>\d*))?')
    match = pat.search(out)
    if not match:
        logmsg('No match.')
        return None
    w = int(match.group('width'))
    h = int(match.group('height'))
    pnmtype = match.group('pnmtype')
    mv = match.group('maxval')
    if mv is None:
        maxval = 1
    else:
        maxval = int(mv)
    logmsg('Type %s, w %i, h %i, maxval %i' % (pnmtype, w, h, maxval))
    return (w, h, pnmtype, maxval)

class MyLogger(object):
    def __init__(self, logger):
        self.logger = logger
    def debug(self, *args):
        return self.logger.debug(' '.join(str(x) for x in args))
    def info(self, *args):
        return self.logger.info(' '.join(str(x) for x in args))
    msg = info

def create_job_logger(job):
    '''
    Create a MyLogger object that writes to a log file within a Job directory.
    '''
    logmsg("getlogger")
    logger = logging.getLogger('job.%i' % job.id)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(job.get_log_file2())
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return MyLogger(logger)

def try_dojob(job, userimage, solve_command, solve_locally):
    print('try_dojob', job, '(sub', job.user_image.submission.id, ')')
    try:
        r = dojob(job, userimage, solve_command=solve_command,
                     solve_locally=solve_locally)
        print('try_dojob', job, 'completed:', r)
        return r
    except OSError as e:
        print('OSError processing job', job)
        print(e)
        import errno
        # Too many open files
        print('e.errno:', e.errno)
        print('errno.EMFILE:', errno.EMFILE)
        if e.errno == errno.EMFILE:
            print('Too many open files -- exiting!')
            sys.exit(-1)
    except IOError as e:
        import errno
        print('Caught IOError')
        print('Errno:', e.errno)
        if e.errno == errno.EMFILE:
            print('Too many open files!')
            sys.exit(-1)
    except:
        print('Caught exception while processing Job', job)
        traceback.print_exc(None, sys.stdout)
        # FIXME -- job.set_status()...
        job.set_end_time()
        job.status = 'F'
        job.save()
        log = create_job_logger(job)
        log.msg('Caught exception while processing Job', job.id)
        log.msg(traceback.format_exc(None))

def dojob(job, userimage, log=None, solve_command=None, solve_locally=None):
    jobdir = job.make_dir()
    #print('Created job dir', jobdir)
    #log = create_job_logger(job)
    #jobdir = job.get_dir()
    if log is None:
        log = create_job_logger(job)
    log.msg('Starting Job processing for', job)
    job.set_start_time()
    job.save()
    #os.chdir(dirnm) - not thread safe (working directory is global)!
    log.msg('Creating directory', jobdir)
    axyfn = 'job.axy'
    axypath = os.path.join(jobdir, axyfn)
    sub = userimage.submission
    log.msg('submission id', sub.id)
    df = userimage.image.disk_file
    img = userimage.image

    # Build command-line arguments for the augment-xylist program, which
    # detects sources in the image and adds processing arguments to the header
    # to produce a "job.axy" file.
    slo,shi = sub.get_scale_bounds()
    # Note, this must match Job.get_wcs_file().
    wcsfile = 'wcs.fits'
    corrfile = 'corr.fits'
    axyflags = []
    axyargs = {
        '--out': axypath,
        '--scale-low': slo,
        '--scale-high': shi,
        '--scale-units': sub.scale_units,
        '--wcs': wcsfile,
        '--corr': corrfile,
        '--rdls': 'rdls.fits',
        '--pixel-error': sub.positional_error,
        '--ra': sub.center_ra,
        '--dec': sub.center_dec,
        '--radius': sub.radius,
        '--downsample': sub.downsample_factor,
        # tuning-up maybe fixed; if not, turn it off with:
        #'--odds-to-tune': 1e9,

        # Other things we might want include...
        # --invert
        # -g / --guess-scale: try to guess the image scale from the FITS headers
        # --crpix-x <pix>: set the WCS reference point to the given position
        # --crpix-y <pix>: set the WCS reference point to the given position
        # -w / --width <pixels>: specify the field width
        # -e / --height <pixels>: specify the field height
        # -X / --x-column <column-name>: the FITS column name
        # -Y / --y-column <column-name>
        }

    if hasattr(img,'sourcelist'):
        # image is a source list; use --xylist
        axyargs['--xylist'] = img.sourcelist.get_fits_path()
        w,h = img.width, img.height
        if sub.image_width:
            w = sub.image_width
        if sub.image_height:
            h = sub.image_height
        axyargs['--width' ] = w
        axyargs['--height'] = h
    else:
        axyargs['--image'] = df.get_path()

    # UGLY
    if sub.parity == 0:
        axyargs['--parity'] = 'pos'
    elif sub.parity == 1:
        axyargs['--parity'] = 'neg'

    if sub.tweak_order == 0:
        axyflags.append('--no-tweak')
    else:
        axyargs['--tweak-order'] = '%i' % sub.tweak_order

    if sub.use_sextractor:
        axyflags.append('--use-sextractor')

    if sub.crpix_center:
        axyflags.append('--crpix-center')

    if sub.invert:
        axyflags.append('--invert')

    cmd = 'augment-xylist '
    for (k,v) in list(axyargs.items()):
        if v:
            cmd += k + ' ' + str(v) + ' '
    for k in axyflags:
        cmd += k + ' '

    log.msg('running: ' + cmd)
    (rtn, out, err) = run_command(cmd)
    if rtn:
        log.msg('out: ' + out)
        log.msg('err: ' + err)
        logmsg('augment-xylist failed: rtn val', rtn, 'err', err)
        raise Exception

    log.msg('created axy file', axypath)
    # shell into compute server...
    logfn = job.get_log_file()
    # the "tar" commands both use "-C" to chdir, and the ssh command
    # and redirect uses absolute paths.

    if solve_locally is not None:

        cmd = (('cd %(jobdir)s && %(solvecmd)s %(jobid)s %(axyfile)s >> ' +
               '%(logfile)s') %
               dict(jobid='job-%s-%i' % (settings.sitename, job.id),
                    solvecmd=solve_locally,
                    axyfile=axyfn, jobdir=jobdir,
                    logfile=logfn))
        log.msg('command:', cmd)
        w = os.system(cmd)
        if not os.WIFEXITED(w):
            log.msg('Solver failed (sent signal?)')
            logmsg('Call to solver failed for job', job.id)
            raise Exception
        rtn = os.WEXITSTATUS(w)
        if rtn:
            log.msg('Solver failed with return value %i' % rtn)
            logmsg('Call to solver failed for job', job.id, 'with return val',
                   rtn)
            raise Exception

        log.msg('Solver completed successfully.')

    else:
        if solve_command is None:
            solve_command = 'ssh -x -T %(sshconfig)s'

        cmd = (('(echo %(jobid)s; '
                'tar cf - --ignore-failed-read -C %(jobdir)s %(axyfile)s) | '
                + solve_command + ' 2>>%(logfile)s | '
                'tar xf - --atime-preserve -m --exclude=%(axyfile)s -C %(jobdir)s '
                '>>%(logfile)s 2>&1') %
               dict(jobid='job-%s-%i' % (settings.sitename, job.id),
                    axyfile=axyfn, jobdir=jobdir,
                    sshconfig=settings.ssh_solver_config,
                    logfile=logfn))
        log.msg('command:', cmd)
        w = os.system(cmd)
        if not os.WIFEXITED(w):
            log.msg('Solver failed (sent signal?)')
            logmsg('Call to solver failed for job', job.id)
            raise Exception
        rtn = os.WEXITSTATUS(w)
        if rtn:
            log.msg('Solver failed with return value %i' % rtn)
            logmsg('Call to solver failed for job', job.id, 'with return val',
                   rtn)
            raise Exception

        log.msg('Solver completed successfully.')

    # Solved?
    wcsfn = os.path.join(jobdir, wcsfile)
    log.msg('Checking for WCS file', wcsfn)
    if os.path.exists(wcsfn):
        log.msg('WCS file exists')
        # Parse the wcs.fits file
        wcs = Tan(wcsfn, 0)
        # Convert to database model...
        tan = TanWCS(crval1=wcs.crval[0], crval2=wcs.crval[1],
                     crpix1=wcs.crpix[0], crpix2=wcs.crpix[1],
                     cd11=wcs.cd[0], cd12=wcs.cd[1],
                     cd21=wcs.cd[2], cd22=wcs.cd[3],
                     imagew=img.width, imageh=img.height)
        tan.save()
        log.msg('Created TanWCS:', tan)

        # Find field's healpix nside and index
        ra, dec, radius = tan.get_center_radecradius()
        nside = anutil.healpix_nside_for_side_length_arcmin(radius*60)
        nside = int(2**round(math.log(nside, 2)))
        nside = max(1, nside)
        healpix = anutil.radecdegtohealpix(ra, dec, nside)
        sky_location, created = SkyLocation.objects.get_or_create(nside=nside, healpix=healpix)
        log.msg('SkyLocation:', sky_location)

        # Find bounds for the Calibration object.
        r0,r1,d0,d1 = wcs.radec_bounds()
        # Find cartesian coordinates
        ra *= math.pi/180
        dec *= math.pi/180
        tempr = math.cos(dec)
        x = tempr*math.cos(ra)
        y = tempr*math.sin(ra)
        z = math.sin(dec)
        r = radius/180*math.pi

        calib = Calibration(raw_tan=tan, ramin=r0, ramax=r1, decmin=d0, decmax=d1,
                            x=x,y=y,z=z,r=r,
                            sky_location=sky_location)
        calib.save()
        log.msg('Created Calibration', calib)
        job.calibration = calib
        job.save() # save calib before adding machine tags
        job.status = 'S'
        job.user_image.add_machine_tags(job)
        job.user_image.add_sky_objects(job)
    else:
        job.status = 'F'
    job.set_end_time()
    job.save()
    log.msg('Finished job', job.id)
    logmsg('Finished job',job.id)
    return job.id

def try_dosub(sub, max_retries):
    subid = sub.id
    try:
        return dosub(sub)
    except DatabaseError as e:
        print('Caught DatabaseError while processing Submission', sub)
        traceback.print_exc(None, sys.stdout)

        # Try...
        django.db.connection.close()
        sub = Submission.objects.get(id=subid)

        if (sub.processing_retries < max_retries):
            print('Retrying processing Submission %s' % str(sub))
            sub.processing_retries += 1
            sub.save()
            return try_dosub(sub, max_retries)
        else:
            print('Submission retry limit reached')
            sub.set_error_message(
                'Caught exception while processing Submission: ' +  str(sub) + '\n'
                + traceback.format_exc(None))
            sub.set_processing_finished()
            sub.save()
            return 'exception'
    except:
        print('Caught exception while processing Submission', sub)
        traceback.print_exc(None, sys.stdout)
        sub.set_error_message(
            'Caught exception while processing Submission: ' +  str(sub) + '\n'
            + traceback.format_exc(None))
        sub.set_processing_finished()
        sub.save()
        logmsg('Caught exception while processing Submission ' + str(sub))
        logmsg('  ' + traceback.format_exc(None))
        return 'exception'

def dosub(sub):
    sub.set_processing_started()
    sub.save()
    print('Submission disk file:', sub.disk_file)

    if sub.disk_file is None:
        logmsg('Sub %i: retrieving URL' % (sub.id), sub.url)
        (fn, headers) = urllib.request.urlretrieve(sub.url)
        logmsg('Sub %i: wrote URL to file' % (sub.id), fn)
        df = DiskFile.from_file(fn, Image.ORIG_COLLECTION)
        logmsg('Created DiskFile', df)
        # Try to split the URL into a filename component and save it
        p = urlparse(sub.url)
        p = p.path
        if p:
            s = p.split('/')
            origname = s[-1]
            sub.original_filename = origname
        df.save()
        sub.disk_file = df
        sub.save()
        logmsg('Saved DiskFile', df)

    else:
        logmsg('uploaded disk file for this submission is ' + str(sub.disk_file))

    df = sub.disk_file
    fn = df.get_path()
    logmsg('DiskFile path ' + fn)

    original_filename = sub.original_filename
    # check if file is a gzipped file
    try:
        with gzip.open(fn) as gzip_file:
            f, tempfn = tempfile.mkstemp()
            os.close(f)
            with open(tempfn, 'wb') as f:
                # should fail on the following line if not a gzip file
                f.write(gzip_file.read())
        df = DiskFile.from_file(tempfn, 'uploaded-gunzip')
        i = original_filename.find('.gz')
        if i != -1:
            original_filename = original_filename[:i]
        logmsg('extracted gzip file %s' % original_filename)
        #fn = tempfn
        fn = df.get_path()
    except:
        # not a gzip file
        pass

    is_tar = False
    try:
        is_tar = tarfile.is_tarfile(fn)
    except:
        pass
    if is_tar:
        logmsg('File %s: tarball' % fn)
        tar = tarfile.open(fn)
        dirnm = tempfile.mkdtemp()
        for tarinfo in tar.getmembers():
            if tarinfo.isfile():
                logmsg('extracting file %s' % tarinfo.name)
                tar.extract(tarinfo, dirnm)
                tempfn = os.path.join(dirnm, tarinfo.name)
                df = DiskFile.from_file(tempfn, 'uploaded-untar')
                # create Image object
                img = get_or_create_image(df)
                # create UserImage object.
                if img:
                    create_user_image(sub, img, tarinfo.name)
        tar.close()
        shutil.rmtree(dirnm, ignore_errors=True)
    else:
        # assume file is single image
        logmsg('File %s: single file' % fn)
        # create Image object
        img = get_or_create_image(df)
        logmsg('File %s: created Image %s' % (fn, str(img)))
        # create UserImage object.
        if img:
            logmsg('File %s: Image id %i' % (fn, img.id))
            uimg = create_user_image(sub, img, original_filename)
            logmsg('Image %i: created UserImage %i' % (img.id, uimg.id))

    sub.set_processing_finished()
    sub.save()
    return sub.id

def create_user_image(sub, img, original_filename):
    pro = get_user_profile(sub.user)
    license, created = License.objects.get_or_create(
        default_license=pro.default_license,
        allow_modifications = sub.license.allow_modifications,
        allow_commercial_use = sub.license.allow_commercial_use,
    )
    comment_receiver = CommentReceiver.objects.create()
    uimg,created = UserImage.objects.get_or_create(
        submission=sub,
        image=img,
        user=sub.user,
        license=license,
        comment_receiver=comment_receiver,
        defaults=dict(original_file_name=original_filename,
                     publicly_visible = sub.publicly_visible))
    if sub.album:
        sub.album.user_images.add(uimg)
    return uimg

def get_or_create_image(df):
    # Is there already an Image for this DiskFile?
    try:
        img = Image.objects.get(disk_file=df, display_image__isnull=False, thumbnail__isnull=False)
    except Image.MultipleObjectsReturned:
        logmsg("multiple found")
        imgs = Image.objects.filter(disk_file=df, display_image__isnull=False, thumbnail__isnull=False)
        for i in range(1,len(imgs)):
            imgs[i].delete()
        img = imgs[0]
    except Image.DoesNotExist:
        # try to create image assume disk file is an image file (png, jpg, etc)
        logmsg('Image database object does not exist; creating')
        img = create_image(df)
        logmsg('img = ' + str(img))
        if img is None:
            # try to create sourcelist image
            img = create_source_list(df)

        if img:
            # cache
            print('Creating thumbnail')
            img.get_thumbnail()
            print('Creating display-sized image')
            img.get_display_image()
            print('Saving image')
            img.save()
        else:
            raise Exception('This file\'s type is not supported.')
    return img


def create_image(df):
    img = None
    try:
        img = Image(disk_file=df)
        # FIXME -- move this code to Image?
        # Convert file to pnm to find its size.
        pnmfn = img.get_pnm_path()
        x = run_pnmfile(pnmfn)
        if x is None:
            raise RuntimeError('Could not find image file size')
        (w, h, pnmtype, maxval) = x
        logmsg('Type %s, w %i, h %i' % (pnmtype, w, h))
        img.width = w
        img.height = h
        img.save()
    except:
        logmsg('file is not an image file: ' + traceback.format_exc())
        img = None
    return img

def create_source_list(df):
    img = None
    fits = None
    source_type = None

    path = df.get_path()
    print('path:', path, type(path))
    
    try:
        # see if disk file is a fits list
        fits = fits_table(str(df.get_path()))
        source_type = 'fits'
    except:
        logmsg('file is not a fits table')
        # otherwise, check to see if it is a text list
        try:
            fitsfn = get_temp_file()

            text_file = open(str(df.get_path()))
            text = text_file.read()
            text_file.close()

            # add x y header
            # potential hack, assumes it doesn't exist...
            text = "# x y\n" + text

            text_table = text_table_fields("", text=text)
            text_table.write_to(fitsfn)
            logmsg("Creating fits table from text list")

            fits = fits_table(fitsfn)
            source_type = 'text'
        except Exception as e:
            #logmsg('Traceback:\n' + traceback.format_exc())
            logmsg('fitsfn: %s' % fitsfn)
            logmsg(e)
            logmsg('file is not a text list')

    if fits:
        try:
            img = SourceList(disk_file=df, source_type=source_type)
            # w = fits.x.max()-fits.x.min()
            # h = fits.y.max()-fits.y.min()
            # w = int(w)
            # h = int(h)
            w = int(math.ceil(fits.x.max()))
            h = int(math.ceil(fits.y.max()))
            logmsg('w %i, h %i' % (w, h))
            if w < 1 or h < 1:
                raise RuntimeError('Source list must contain POSITIVE x,y coordinates')
            img.width = w
            img.height = h
            img.save()
        except Exception as e:
            logmsg(e)
            img = None
            raise e

    return img

## DEBUG
def sub_callback(result):
    print('Submission callback: Result:', result)
def job_callback(result):
    print('Job callback: Result:', result)


def main(dojob_nthreads, dosub_nthreads, refresh_rate, max_sub_retries,
         solve_command, solve_locally):
    dojob_pool = None
    dosub_pool = None
    if dojob_nthreads > 1:
        print('Processing jobs with %d threads' % dojob_nthreads)
        dojob_pool = multiprocessing.Pool(processes=dojob_nthreads)
    if dosub_nthreads > 1:
        print('Processing submissions with %d threads' % dosub_nthreads)
        dosub_pool = multiprocessing.Pool(processes=dosub_nthreads)

    print('Refresh rate: %.1f seconds' % refresh_rate)
    print('Submission processing retry limit: %d' % max_sub_retries)

    # Find Submissions that have been started but not finished;
    # reset the start times to null.
    oldsubs = Submission.objects.filter(processing_started__isnull=False,
                                        processing_finished__isnull=True)
    for sub in oldsubs:
        print('Resetting the processing status of', sub)
        sub.processing_started = None
        sub.save()

    oldjobs = Job.objects.filter(Q(end_time__isnull=True),
                                 Q(start_time__isnull=False) |
                                 Q(queued_time__isnull=False))
    #for job in oldjobs:
    #    #print('Resetting the processing status of', job)
    #    #job.start_time = None
    #    #job.save()
    # FIXME -- really?
    oldjobs.delete()

    subresults = []
    jobresults = []

    #
    me = ProcessSubmissions(pid=os.getpid())
    me.set_watchdog()
    me.save()

    lastsubs = []
    lastjobs = []

    while True:
        me.set_watchdog()
        me.save()

        print()

        #print('Checking for new Submissions')
        newsubs = Submission.objects.filter(processing_started__isnull=True)
        if newsubs.count():
            print('Found', newsubs.count(), 'unstarted Submissions:', [s.id for s in newsubs])

        #print('Checking for UserImages without Jobs')
        all_user_images = UserImage.objects.annotate(job_count=Count('jobs'))
        newuis = all_user_images.filter(job_count=0)
        if newuis.count():
            #print('Found', len(newuis), 'UserImages without Jobs:', [u.id for u in newuis])
            #print('Found', len(newuis), 'UserImages without Jobs.')
            print('Jobs queued:', len(newuis))

        runsubs = me.subs.filter(finished=False)
        if subresults != lastsubs:
            print('Submissions running:', len(subresults))
            lastsubs = subresults
        for sid,res in subresults:
            print('  Submission id', sid, 'ready:', res.ready(),)
            if res.ready():
                subresults.remove((sid,res))
                print('success:', res.successful(),)

                qs = runsubs.get(submission__id=sid)
                qs.finished = True
                qs.success = res.successful()
                qs.save()

                if res.successful():
                    print('result:', res.get(),)
            print()

        runjobs = me.jobs.filter(finished=False)
        if jobresults != lastjobs:
            print('Jobs running:', len(jobresults))
            lastjobs = jobresults
        for jid,res in jobresults:
            print('  Job id', jid, 'ready:', res.ready(),)
            if res.ready():
                jobresults.remove((jid,res))
                print('success:', res.successful(),)

                qj = runjobs.get(job__id=jid)
                qj.finished = True
                qj.success = res.successful()
                qj.save()

                try:
                    job = Job.objects.get(id=jid)
                    print('Job:', job)
                    print('  status:', job.status)
                    print('  error message:', job.error_message)
                    #logfn = job.get_log_file()
                    print('  log file tail:')
                    print(job.get_log_tail(nlines=10))
                    
                except:
                    print('exception getting job')
                
                if res.successful():
                    print('result:', res.get(),)
            print()
        if len(jobresults):
            print('Still waiting for', len(jobresults), 'Jobs')

        if (len(newsubs) + len(newuis)) == 0:
            time.sleep(refresh_rate)
            continue

        # FIXME -- order by user, etc

        ## HACK -- order 'newuis' to do the newest ones first... helpful when there
        # is a big backlog.
        newuis = newuis.order_by('-submission__submitted_on')

        for sub in newsubs:
            print('Enqueuing submission:', str(sub))
            sub.set_processing_started()
            sub.save()

            qs = QueuedSubmission(procsub=me, submission=sub)
            qs.save()

            if dosub_pool:
                res = dosub_pool.apply_async(
                    try_dosub,
                    (sub, max_sub_retries),
                    callback=sub_callback
                )
                subresults.append((sub.id, res))
            else:
                try_dosub(sub, max_sub_retries)


        if dojob_pool:
            n_add = dojob_nthreads - len(jobresults)
            if n_add <= 0:
                # Already full!
                continue
            # Queue some new ones -- randomly select from waiting users
            newuis = list(newuis)
            start_newuis = []
            import numpy as np
            from collections import Counter
            print('Need to start', n_add, 'jobs;', len(newuis), 'eligible uis')
            while n_add > 0:
                cusers = Counter([u.user for u in newuis])
                print('Jobs queued:', len(newuis), 'by', len(cusers), 'users; top:')
                for k,user in cusers.most_common(5):
                    try:
                        print('  ', k, user, user.get_profile().display_name)
                    except:
                        print('  ', k, user)
                users = list(cusers.keys())
                print(len(users), 'eligible users')
                if len(users) == 0:
                    break
                iu = np.random.randint(len(users))
                user = users[iu]
                print('Selected user', user)
                for ui in newuis:
                    if ui.user == user:
                        print('Selected ui', ui)
                        newuis.remove(ui)
                        start_newuis.append(ui)
                        n_add -= 1
                        break
            newuis = start_newuis

        for userimage in newuis:
            job = Job(user_image=userimage)
            job.set_queued_time()
            job.save()

            qj = QueuedJob(procsub=me, job=job)
            qj.save()

            if dojob_pool:
                res = dojob_pool.apply_async(try_dojob, (job, userimage, solve_command, solve_locally),
                                             callback=job_callback)
                jobresults.append((job.id, res))
            else:
                dojob(job, userimage, solve_command=solve_command, solve_locally=solve_locally)

if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser()
    parser.add_option('--jobthreads', '-j', dest='jobthreads', type='int',
                      default=3, help='Set the number of threads to process jobs')
    parser.add_option('--subthreads', '-s', dest='subthreads', type='int',
                      default=2, help='Set the number of threads to process submissions')
    parser.add_option('--maxsubretries', '-m', dest='maxsubretries', type='int',
                      default=20, help='Set the maximum number of times to retry processing a submission')
    parser.add_option('--refreshrate', '-r', dest='refreshrate', type='float',
                      default=5, help='Set how often to check for new jobs and submissions (in seconds)')

    parser.add_option('--solve-command', default=None,
                      help='Command to run instead of ssh to actually solve image, eg "testscript-astro"')

    parser.add_option('--solve-locally',
                      help='Command to run astrometry-engine on this machine, not via ssh')

    opt,args = parser.parse_args()

    main(opt.jobthreads, opt.subthreads, opt.refreshrate, opt.maxsubretries,
         opt.solve_command, opt.solve_locally)
