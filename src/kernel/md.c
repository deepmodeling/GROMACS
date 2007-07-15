/*
 * $Id$
 * 
 *                This source code is part of
 * 
 *                 G   R   O   M   A   C   S
 * 
 *          GROningen MAchine for Chemical Simulations
 * 
 *                        VERSION 3.2.0
 * Written by David van der Spoel, Erik Lindahl, Berk Hess, and others.
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team,
 * check out http://www.gromacs.org for more information.

 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 * 
 * If you want to redistribute modifications, please consider that
 * scientific software is very special. Version control is crucial -
 * bugs must be traceable. We will be happy to consider code for
 * inclusion in the official distribution, but derived work must not
 * be called official GROMACS. Details are found in the README & COPYING
 * files - if they are missing, get the official version at www.gromacs.org.
 * 
 * To help us fund GROMACS development, we humbly ask that you cite
 * the papers on the package - you can find them in the top README file.
 * 
 * For more info, check our website at http://www.gromacs.org
 * 
 * And Hey:
 * Gallium Rubidium Oxygen Manganese Argon Carbon Silicon
 */
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <signal.h>
#include <stdlib.h>
#include "typedefs.h"
#include "smalloc.h"
#include "sysstuff.h"
#include "vec.h"
#include "statutil.h"
#include "vcm.h"
#include "mdebin.h"
#include "nrnb.h"
#include "calcmu.h"
#include "index.h"
#include "vsite.h"
#include "update.h"
#include "ns.h"
#include "trnio.h"
#include "xtcio.h"
#include "mdrun.h"
#include "confio.h"
#include "network.h"
#include "pull.h"
#include "xvgr.h"
#include "physics.h"
#include "names.h"
#include "xmdrun.h"
#include "disre.h"
#include "orires.h"
#include "dihre.h"
#include "pppm.h"
#include "pme.h"
#include "mdatoms.h"
#include "repl_ex.h"
#include "qmmm.h"
#include "mpelogging.h"
#include "domdec.h"
#include "partdec.h"
#include "coulomb.h"
#include "constr.h"
#include "compute_io.h"

#ifdef GMX_MPI
#include <mpi.h>
#endif

/* The following two variables and the signal_handler function
 * is used from pme.c as well 
 */
extern bool bGotTermSignal, bGotUsr1Signal;

static RETSIGTYPE signal_handler(int n)
{
  switch (n) {
  case SIGTERM:
    bGotTermSignal = TRUE;
    break;
  case SIGUSR1:
    bGotUsr1Signal = TRUE;
    break;
  }
}

static void send_inputrec(t_commrec *cr,
			  t_inputrec *inputrec,int nChargePerturbed)
{
  int dest;
  
  if (MASTER(cr) && cr->npmenodes > 0) {
    if (!EEL_PME(inputrec->coulombtype))
      gmx_fatal(FARGS,"Separate PME nodes have been selected with %s electrostatics",EELTYPE(inputrec->coulombtype));
    for(dest=0; dest<cr->nnodes; dest++) {
      if (gmx_pmeonlynode(cr,dest)) {
#ifdef GMX_MPI
	/* dest is a PME only node */
	/* Send the inputrec to a PME node */
	MPI_Send(inputrec,sizeof(t_inputrec),MPI_BYTE,
		 dest,0,cr->mpi_comm_mysim);
	/* Tell if we need to do PME with free energy */
	MPI_Send(&nChargePerturbed,sizeof(int),MPI_BYTE,
		 dest,0,cr->mpi_comm_mysim);
#endif
      }
    }
  }
}

static void receive_inputrec(t_commrec *cr,
			     t_inputrec *inputrec,int *nChargePerturbed)
{
#ifdef GMX_MPI
  MPI_Recv(inputrec,sizeof(t_inputrec),MPI_BYTE,
	   0,0,cr->mpi_comm_mysim,
	   MPI_STATUS_IGNORE);
  MPI_Recv(nChargePerturbed,sizeof(int),MPI_BYTE,
	   0,0,cr->mpi_comm_mysim,
	   MPI_STATUS_IGNORE);
#endif
}

void mdrunner(t_commrec *cr,int nfile,t_filenm fnm[],
	      bool bVerbose,bool bCompact,
	      ivec ddxyz,int dd_node_order,real rdd,
	      char *ddcsx,char *ddcsy,char *ddcsz,
	      int nstepout,t_edsamyn *edyn,int repl_ex_nst,int repl_ex_seed,
	      unsigned long Flags)
{
  double     nodetime=0,realtime;
  t_inputrec *inputrec;
  t_state    *state=NULL;
  rvec       *buf=NULL,*f=NULL;
  real       tmpr1,tmpr2;
  real       *ener=NULL;
  t_nrnb     *nrnb;
  t_topology *top=NULL;
  t_groups   *grps=NULL;
  t_graph    *graph=NULL;
  t_mdatoms  *mdatoms=NULL;
  t_forcerec *fr=NULL;
  t_fcdata   *fcd=NULL;
  real       ewaldcoeff=0;
  gmx_pme_t  *pmedata=NULL;
  time_t     start_t=0;
  gmx_vsite_t *vsite=NULL;
  int        i,m,nChargePerturbed=0,status,nalloc;
  char       *gro;
  gmx_wallcycle_t wcycle;

  if ((ddxyz[XX]!=1 || ddxyz[YY]!=1 || ddxyz[ZZ]!=1)) {
    cr->dd = init_domain_decomposition(stdlog,cr,ddxyz,rdd,
				       Flags & MD_DLB,ddcsx,ddcsy,ddcsz);
    
    make_dd_communicators(stdlog,cr,dd_node_order);
  } else {
    if (cr->npmenodes > 0)
      gmx_fatal(FARGS,
		"Can only use seperate PME nodes with domain decomposition\n");
    cr->duty = (DUTY_PP | DUTY_PME);
  }

  wcycle = wallcycle_init(stdlog,cr);

  snew(inputrec,1);
  snew(nrnb,1);
  if (cr->duty & DUTY_PP) {
    /* Initiate everything (snew sets to zero!) */
    snew(ener,F_NRE);
    snew(fcd,1);
    snew(top,1);
    snew(grps,1);
    snew(state,1);

    if (bVerbose && MASTER(cr)) 
      fprintf(stderr,"Getting Loaded...\n");
    
    if (PAR(cr)) {
      /* The master thread on the master node reads from disk, 
       * then dsitributes everything to the other processors.
       */
      init_parallel(stdlog,ftp2fn(efTPX,nfile,fnm),cr,
		    inputrec,top,state,
		    MASTER(cr) ? LIST_SCALARS | LIST_INPUTREC : 0);
      
      if (!(EI_TPI(inputrec->eI) || DOMAINDECOMP(cr))) {
	split_system(stdlog,inputrec,state,cr,top);
      }
    }
    else {
      /* Read a file for a single processor */
      init_single(stdlog,inputrec,ftp2fn(efTPX,nfile,fnm),top,state);
    }
    
    if (bVerbose && MASTER(cr))
      fprintf(stderr,"Loaded with Money\n\n");
    
    /* For domain decomposition we allocate dynamically
     * in dd_partition_system.
     */
    if (!DOMAINDECOMP(cr)) {
      snew(buf,top->atoms.nr);
      snew(f,top->atoms.nr);
    }
    
    /* Index numbers for parallellism... */
    top->idef.nodeid = cr->nodeid;
    
    /* Group stuff (energies etc) */
    init_groups(stdlog,&top->atoms,&(inputrec->opts),grps);
    /* Copy the cos acceleration to the groups struct */
    grps->cosacc.cos_accel = inputrec->cos_accel;
    
    /* Periodicity stuff */  
    if (inputrec->ePBC != epbcNONE && !inputrec->bPeriodicMols &&
	!DOMAINDECOMP(cr)) {
      graph = mk_graph(&(top->idef),top->atoms.nr,FALSE,FALSE);
      if (gmx_debug_at) 
	p_graph(debug,"Initial graph",graph);
    }
    else
      graph = NULL;
    
    /* Distance Restraints */
    init_disres(stdlog,top->idef.il[F_DISRES].nr,top->idef.il[F_DISRES].iatoms,
		top->idef.iparams,inputrec,cr->ms,fcd);
    
    if (top->idef.il[F_ORIRES].nr) {
      if (DOMAINDECOMP(cr))
	gmx_fatal(FARGS,"Orientation restraints do not work with domain decomposition");
      else
	/* Orientation restraints */
	init_orires(stdlog,
		    top->idef.il[F_ORIRES].nr,top->idef.il[F_ORIRES].iatoms,
		    top->idef.iparams,state->x,&top->atoms,inputrec,cr->ms,
		    &(fcd->orires));
    }
    
    /* Dihedral Restraints */
    init_dihres(stdlog,top->idef.il[F_DIHRES].nr,top->idef.il[F_DIHRES].iatoms,
		top->idef.iparams,inputrec,fcd);
    
    /* Initiate forcerecord */
    fr = mk_forcerec();
    init_forcerec(stdlog,fr,fcd,inputrec,top,cr,state->box,FALSE,
		  opt2fn("-table",nfile,fnm),opt2fn("-tablep",nfile,fnm),
		  opt2fn("-tableb",nfile,fnm),FALSE);
    fr->bSepDVDL = ((Flags & MD_SEPPOT) == MD_SEPPOT);
    
    /* Initialize QM-MM */
    if(fr->bQMMM)
      init_QMMMrec(cr,state->box,top,inputrec,fr);
    
    /* Initialize the mdatoms structure.
     * mdatoms is not filled with atom data,
     * as this can not be done now with domain decomposition.
     */
    mdatoms = init_mdatoms(stdlog,&top->atoms,inputrec->efep!=efepNO);
    
    /* Initialize the virtual site communication */
    vsite = init_vsite(cr,top);

    /* Make molecules whole at start of run */
    if (fr->ePBC != epbcNONE)  {
      do_pbc_first(stdlog,state->box,fr,graph,state->x);
    }

    /* Initiate PPPM if necessary */
    if (fr->eeltype == eelPPPM) {
      if (mdatoms->nChargePerturbed)
	gmx_fatal(FARGS,"Free energy with %s is not implemented",
		  eel_names[fr->eeltype]);
      status = gmx_pppm_init(stdlog,cr,FALSE,TRUE,state->box,
			     getenv("GMXGHAT"),inputrec);
      if (status != 0)
	gmx_fatal(FARGS,"Error %d initializing PPPM",status);
    }

    send_inputrec(cr,inputrec,mdatoms->nChargePerturbed);
  }

  /* Initiate PME if necessary */
  /* either on all nodes (if epmePMEANDPP is TRUE) 
   * or on dedicated PME nodes (if epmePMEONLY is TRUE) */
  if (!(cr->duty & DUTY_PP) || EEL_PME(fr->eeltype)) {
    if (cr->duty & DUTY_PME) {
      if (cr->duty & DUTY_PP) {
	ewaldcoeff = fr->ewaldcoeff;
	pmedata = &fr->pmedata;
	nChargePerturbed = mdatoms->nChargePerturbed;
      } else {
	receive_inputrec(cr,inputrec,&nChargePerturbed);
	ewaldcoeff = calc_ewaldcoeff(inputrec->rcoulomb, inputrec->ewald_rtol);
	snew(pmedata,1);
      }
      
      status = gmx_pme_init(pmedata,cr,inputrec,
			    top ? top->atoms.nr : 0,nChargePerturbed);
      if (status != 0)
	gmx_fatal(FARGS,"Error %d initializing PME",status);
    }
  }
  
  switch (inputrec->eI) {
  case eiMD:
  case eiSD:
  case eiBD:
    /* Turn on signal handling on all nodes */
    /*
     * (A user signal from the PME nodes (if any)
     * is communicated to the PP nodes.
     */
    if (getenv("GMX_NO_TERM") == NULL) {
      if (debug)
	fprintf(debug,"Installing signal handler for SIGTERM\n");
      signal(SIGTERM,signal_handler);
    }
    if (getenv("GMX_NO_USR1") == NULL) {
      if (debug)
	fprintf(debug,"Installing signal handler for SIGUSR1\n");
      signal(SIGUSR1,signal_handler);
    }
    break;
  default:
    /* No signal handling */
    break;
  }

  if (cr->duty & DUTY_PP) {
    /* Now do whatever the user wants us to do (how flexible...) */
    switch (inputrec->eI) {
    case eiMD:
    case eiSD:
    case eiBD:
      start_t=do_md(stdlog,cr,nfile,fnm,
		    bVerbose,bCompact,
		    ddxyz,ddcsx,ddcsy,ddcsz,
		    vsite,
		    nstepout,inputrec,grps,top,ener,fcd,state,f,buf,
		    mdatoms,nrnb,wcycle,graph,edyn,fr,
		    repl_ex_nst,repl_ex_seed,
		    Flags);
      break;
    case eiCG:
      start_t=do_cg(stdlog,nfile,fnm,inputrec,top,grps,
		    state,f,buf,mdatoms,ener,fcd,
		    nrnb,wcycle,bVerbose,vsite,
		    cr,graph,fr);
      break;
    case eiLBFGS:
      start_t=do_lbfgs(stdlog,nfile,fnm,inputrec,top,grps,
		       state,f,buf,mdatoms,ener,fcd,
		       nrnb,wcycle,bVerbose,vsite,
		       cr,graph,fr);
      break;
    case eiSteep:
      start_t=do_steep(stdlog,nfile,fnm,inputrec,top,grps,
		       state,f,buf,mdatoms,ener,fcd,
		       nrnb,wcycle,bVerbose,vsite,
		       cr,graph,fr);
    break;
    case eiNM:
      start_t=do_nm(stdlog,cr,nfile,fnm,
		    bVerbose,bCompact,nstepout,inputrec,grps,
		    top,ener,fcd,state,f,buf,
		    mdatoms,nrnb,wcycle,vsite,graph,edyn,fr);
      break;
    case eiTPI:
    case eiTPIC:
      start_t=do_tpi(stdlog,nfile,fnm,inputrec,top,grps,
		     state,f,buf,mdatoms,ener,fcd,
		     nrnb,wcycle,bVerbose,
		     cr,graph,fr);
      break;
    default:
      gmx_fatal(FARGS,"Invalid integrator (%d)...\n",inputrec->eI);
    }
  } else {
    /* do PME only */
    gmx_pmeonly(*pmedata,cr,nrnb,wcycle,ewaldcoeff,FALSE);
  }
 
  /* Some timing stats */  
  if (MASTER(cr)) {
    realtime=difftime(time(NULL),start_t);
    if ((nodetime=node_time()) == 0)
      nodetime=realtime;
  }
  else 
    realtime=0;

  wallcycle_stop(wcycle,ewcRUN);
    
  /* Finish up, write some stuff
   * if rerunMD, don't write last frame again 
   */
  finish_run(stdlog,cr,ftp2fn(efSTO,nfile,fnm),
	     top,inputrec,nrnb,wcycle,nodetime,realtime,inputrec->nsteps,
	     EI_DYNAMICS(inputrec->eI) && !MULTISIM(cr));
  
  /* Does what it says */  
  print_date_and_time(stdlog,cr->nodeid,"Finished mdrun");
}

time_t do_md(FILE *log,t_commrec *cr,int nfile,t_filenm fnm[],
	     bool bVerbose,bool bCompact,
	     ivec ddxyz,char *ddcsx,char *ddcsy,char *ddcsz,
	     gmx_vsite_t *vsite,
	     int stepout,t_inputrec *inputrec,t_groups *grps,
	     t_topology *top_global,
	     real ener[],t_fcdata *fcd,
	     t_state *state_global,rvec f[],
	     rvec buf[],t_mdatoms *mdatoms,
	     t_nrnb *nrnb,gmx_wallcycle_t wcycle,
	     t_graph *graph,t_edsamyn *edyn,t_forcerec *fr,
	     int repl_ex_nst,int repl_ex_seed,
	     unsigned long Flags)
{
  t_mdebin   *mdebin;
  int        fp_ene=0,fp_trn=0,fp_xtc=0,step,step_rel;
  FILE       *fp_dgdl=NULL,*fp_field=NULL;
  time_t     start_t;
  real       t,t0,lam0;
  bool       bNS,bSimAnn,bStopCM,bRerunMD,bNotLastFrame=FALSE,
             bFirstStep,bLastStep,bNEMD,do_log,do_verbose,bRerunWarnNoV=TRUE,
	     bForceUpdate=FALSE,bX,bV,bF,bXTC,bMasterState;
  tensor     force_vir,shake_vir,total_vir,pres,ekin;
  int        i,m,status;
  rvec       mu_tot;
  t_vcm      *vcm;
  t_trxframe rerun_fr;
  gmx_repl_ex_t *repl_ex=NULL;
  /* A boolean (disguised as a real) to terminate mdrun */  
  real       terminate=0;

  t_topology *top;
  t_state    *state=NULL;
  rvec       *f_global=NULL;
  gmx_stochd_t sd=NULL;
  gmx_constr_t constr=NULL;

  /* XMDRUN stuff: shell, general coupling etc. */
  bool        bFFscan;
  int         nshell,count,nconverged=0;
  t_shell     *shells=NULL;
  real        timestep=0;
  double      tcount=0;
  bool        bHaveConstr=FALSE,bShell_FlexCon,bIonize=FALSE,bGlas=FALSE;
  bool        bTCR=FALSE,bConverged=TRUE,bOK,bExchanged;
  real        temp0,mu_aver=0,fmax;
  int         a0,a1,gnx,ii;
  atom_id     *grpindex;
  char        *grpname;
  t_coupl_rec *tcr=NULL;
  rvec        *xcopy=NULL,*vcopy=NULL;
  matrix      boxcopy,lastbox;
  /* End of XMDRUN stuff */

  /* Check for special mdrun options */
  bRerunMD = (Flags & MD_RERUN)  == MD_RERUN;
  bIonize  = (Flags & MD_IONIZE) == MD_IONIZE;
  bGlas    = (Flags & MD_GLAS)   == MD_GLAS;
  bFFscan  = (Flags & MD_FFSCAN) == MD_FFSCAN;

  if (bRerunMD || bFFscan)
    inputrec->nstxtcout = 0;

  /* Initial values */
  init_md(cr,inputrec,&t,&t0,&state_global->lambda,&lam0,
	  nrnb,top_global,&sd,&constr,
	  nfile,fnm,&fp_trn,&fp_xtc,&fp_ene,&fp_dgdl,&fp_field,&mdebin,grps,
	  force_vir,shake_vir,mu_tot,&bNEMD,&bSimAnn,&vcm);
  debug_gmx();

  if (PARTDECOMP(cr)) {
    pd_at_range(cr,&a0,&a1);
  } else {
    a0 = 0;
    a1 = top_global->atoms.nr;
  }

  if (inputrec->ePull != epullNO) {
    /* Initialize pull code */
    init_pull(log,inputrec,nfile,fnm,
	      state_global->x,&top_global->atoms,state_global->box,cr,a0,a1);
    if (inputrec->ePull == epullCONSTRAINT)
      bHaveConstr = TRUE;
  }

  {
    double io = compute_io(inputrec,&top_global->atoms,mdebin->ebin->nener,1);
    if ((io > 2000) && MASTER(cr))
      fprintf(stderr,
	      "\nWARNING: This run will generate roughly %.0f Mb of data\n\n",
	      io);
  }
  
  if (DOMAINDECOMP(cr)) {
    /* Set overallocation to avoid frequent reallocation of arrays */
    set_over_alloc_dd(TRUE);

    set_dd_parameters(stdlog,cr->dd,top_global,inputrec,fr);

    dd_make_reverse_top(stdlog,cr->dd,top_global,vsite,
			EI_DYNAMICS(inputrec->eI),inputrec->coulombtype);

    top = dd_init_local_top(top_global);

    snew(state,1);
    init_state(state,0,state_global->ngtc);
    state->natoms = state_global->natoms;
    state->flags  = state_global->flags;

    if (DDMASTER(cr->dd) && inputrec->nstfout)
      snew(f_global,state->natoms);

    setup_dd_grid(stdlog,cr->dd);

    if (DDMASTER(cr->dd) && vsite && !inputrec->bContinuation)
      construct_vsites(log,vsite,
		       state_global->x,nrnb,inputrec->delta_t,NULL,
		       &top_global->idef,inputrec->ePBC,TRUE,NULL,
		       NULL,state_global->box);

    dd_partition_system(stdlog,inputrec->init_step,cr,TRUE,
			state_global,top_global,inputrec,
			state,&f,&buf,mdatoms,top,fr,vsite,constr,
			nrnb,wcycle,FALSE);
  } else {
    top = top_global;
    state = state_global;
    f_global = f;

    atoms2md(&top->atoms,inputrec,top->idef.il[F_ORIRES].nr,0,NULL,a0,a1-a0,
	     mdatoms);
  }
  update_mdatoms(mdatoms,state->lambda);

  /* Initialize constraints */
  if (constr) {
    if (!DOMAINDECOMP(cr))
      set_constraints(stdlog,constr,top,inputrec,mdatoms,NULL);
    bHaveConstr = TRUE;
  }

  init_edsam(stdlog,top,inputrec,mdatoms,mdatoms->start,mdatoms->homenr,cr,
	     edyn);
  if (ed_constraints(edyn))
    bHaveConstr = TRUE;
    
  /* Check for polarizable models */
  shells = init_shells(log,cr,&top->idef,mdatoms,&nshell);

  /* Do we need to minimize at every MD step? */
  bShell_FlexCon = (shells || n_flexible_constraints(constr) > 0);

  /* Initialize the essential dynamics sampling */
  do_first_edsam(stdlog,top,mdatoms,mdatoms->start,mdatoms->homenr,cr,
		 state->x,state->box,edyn,bHaveConstr);
  if (ed_constraints(edyn))
    bHaveConstr = TRUE;
  
  gnx = top->blocks[ebMOLS].nr;
  snew(grpindex,gnx);
  for(i=0; (i<gnx); i++)
    grpindex[i] = i;

  /* Check whether we have to GCT stuff */
  bTCR = ftp2bSet(efGCT,nfile,fnm);
  if (MASTER(cr) && bTCR)
    fprintf(stderr,"Will do General Coupling Theory!\n");

  if (repl_ex_nst > 0 && MASTER(cr))
    repl_ex = init_replica_exchange(log,cr->ms,state_global,inputrec,
				    repl_ex_nst,repl_ex_seed);
  
  if (bHaveConstr && !inputrec->bContinuation && !bRerunMD)
    do_shakefirst(log,constr,inputrec,mdatoms,state,buf,f,
		  graph,cr,nrnb,grps,fr,top,edyn);
  debug_gmx();

  /* Compute initial EKin for all.. */
  if (grps->cosacc.cos_accel == 0)
    calc_ke_part(state->v,&(inputrec->opts),mdatoms,grps,nrnb,state->lambda);
  else
    calc_ke_part_visc(state->box,state->x,state->v,&(inputrec->opts),
		      mdatoms,grps,nrnb,state->lambda);
  debug_gmx();

  if (PAR(cr)) 
  {
    GMX_MPE_LOG(ev_global_stat_start);
       
    global_stat(log,cr,ener,force_vir,shake_vir,mu_tot,
		inputrec,grps,constr,vcm,&terminate);

    GMX_MPE_LOG(ev_global_stat_finish);
  }
  debug_gmx();
  
  /* Calculate the initial half step temperature */
  temp0 = sum_ekin(TRUE,&(inputrec->opts),grps,ekin,NULL);

  debug_gmx();
   
  /* Initiate data for the special cases */
  if (bFFscan) {
    snew(xcopy,state->natoms);
    snew(vcopy,state->natoms);
    for(ii=0; (ii<state->natoms); ii++) {
      copy_rvec(state->x[ii],xcopy[ii]);
      copy_rvec(state->v[ii],vcopy[ii]);
    }
    copy_mat(state->box,boxcopy);
  } 

  if (MASTER(cr)) {
    if (bHaveConstr && !inputrec->bContinuation)
      fprintf(log,
	      "RMS relative constraint deviation after constraining: %.2e\n",
	      constr_rmsd(constr,FALSE));
    fprintf(log,"Initial temperature: %g K\n",temp0);
    if (bRerunMD) {
      fprintf(stderr,"starting md rerun '%s', reading coordinates from"
	      " input trajectory '%s'\n\n",
	      *(top->name),opt2fn("-rerun",nfile,fnm));
      if (bVerbose)
	fprintf(stderr,"Calculated time to finish depends on nsteps from "
		"run input file,\nwhich may not correspond to the time "
		"needed to process input trajectory.\n\n");
    } else {
      fprintf(stderr,"starting mdrun '%s'\n%d steps, %8.1f ps.\n",
	      *(top->name),inputrec->nsteps,inputrec->nsteps*inputrec->delta_t);
    }
    fprintf(log,"\n");
  }

  /* Write start time */
  start_t=print_date_and_time(log,cr->nodeid,"Started mdrun");
  wallcycle_start(wcycle,ewcRUN);
  if (log)
    fprintf(log,"\n");

  /* Set the node time counter to 0 after initialisation */
  start_time();
  debug_gmx();
  /***********************************************************
   *
   *             Loop over MD steps 
   *
   ************************************************************/
  
  /* if rerunMD then read coordinates and velocities from input trajectory */
  if (bRerunMD) {
    if (getenv("GMX_FORCE_UPDATE"))
      bForceUpdate = TRUE;

    bNotLastFrame = read_first_frame(&status,opt2fn("-rerun",nfile,fnm),
				     &rerun_fr,TRX_NEED_X | TRX_READ_V);
    if (rerun_fr.natoms != mdatoms->nr)
      gmx_fatal(FARGS,"Number of atoms in trajectory (%d) does not match the "
		  "run input file (%d)\n",rerun_fr.natoms,mdatoms->nr);
    if (inputrec->ePBC != epbcNONE) {
      if (!rerun_fr.bBox)
	gmx_fatal(FARGS,"Rerun trajectory frame step %d time %f does not contain a box, while pbc is used",rerun_fr.step,rerun_fr.time);
      if (max_cutoff2(inputrec->ePBC,rerun_fr.box) < sqr(fr->rlistlong))
	gmx_fatal(FARGS,"Rerun trajectory frame step %d time %f has too small box dimensions",rerun_fr.step,rerun_fr.time);
    }
  }

  /* loop over MD steps or if rerunMD to end of input trajectory */
  bFirstStep = TRUE;
  bLastStep = FALSE;
  bExchanged = FALSE;
  step = inputrec->init_step;
  step_rel = 0;

  while ((!bRerunMD && (step_rel <= inputrec->nsteps)) ||  
	 (bRerunMD && bNotLastFrame)) {

    GMX_MPE_LOG(ev_timestep1);

    if (bRerunMD) {
      if (rerun_fr.bStep) {
	step = rerun_fr.step;
	step_rel = step - inputrec->init_step;
      }
      if (rerun_fr.bTime)
	t = rerun_fr.time;
      else
	t = step;
    } else {
      bLastStep = (step_rel == inputrec->nsteps);

      t = t0 + step*inputrec->delta_t;
    }
    
    do_log = do_per_step(step,inputrec->nstlog) || bLastStep;
    do_verbose = bVerbose && (step % stepout == 0 || bLastStep);

    if (inputrec->efep != efepNO) {
      if (bRerunMD && rerun_fr.bLambda && (inputrec->delta_lambda!=0))
	state->lambda = rerun_fr.lambda;
      else
	state->lambda = lam0 + step*inputrec->delta_lambda;
    }
    
    if (bSimAnn) 
      update_annealing_target_temp(&(inputrec->opts),t);
    
    if (bRerunMD) {
      for(i=0; i<mdatoms->nr; i++)
	copy_rvec(rerun_fr.x[i],state->x[i]);
      if (rerun_fr.bV)
	for(i=0; i<mdatoms->nr; i++)
	  copy_rvec(rerun_fr.v[i],state->v[i]);
      else {
	for(i=0; i<mdatoms->nr; i++)
	    clear_rvec(state->v[i]);
	if (bRerunWarnNoV) {
	  fprintf(stderr,"\nWARNING: Some frames do not contain velocities.\n"
		  "         Ekin, temperature and pressure are incorrect,\n"
		  "         the virial will be incorrect when constraints are present.\n"
		  "\n");
	  bRerunWarnNoV = FALSE;
	}
      }
      copy_mat(rerun_fr.box,state->box);
      
      /* for rerun MD always do Neighbour Searching */
      bNS = ((inputrec->nstlist!=0) || bFirstStep);
    } else {
      /* Determine whether or not to do Neighbour Searching */
      bNS = ((inputrec->nstlist && (step % inputrec->nstlist==0 || bExchanged))
	     || bFirstStep);
    }
    
    /* Stop Center of Mass motion */
    bStopCM = do_per_step(step,abs(inputrec->nstcomm));

    /* Copy back starting coordinates in case we're doing a forcefield scan */
    if (bFFscan) {
      for(ii=0; (ii<state->natoms); ii++) {
	copy_rvec(xcopy[ii],state->x[ii]);
	copy_rvec(vcopy[ii],state->v[ii]);
      }
      copy_mat(boxcopy,state->box);
    }

    if (vsite) {
	wallcycle_start(wcycle,ewcVSITECONSTR);
	if (graph) {
	/* Following is necessary because the graph may get out of sync
	 * with the coordinates if we only have every N'th coordinate set
	 */
	if (bRerunMD || bExchanged)
	  mk_mshift(log,graph,fr->ePBC,state->box,state->x);
	shift_self(graph,state->box,state->x);
      }
      construct_vsites(log,vsite,state->x,nrnb,inputrec->delta_t,state->v,
		       &top->idef,fr->ePBC,fr->bMolPBC,graph,cr,state->box);
      
      if (graph)
	unshift_self(graph,state->box,state->x);
      wallcycle_stop(wcycle,ewcVSITECONSTR);
    }
    debug_gmx();

    if (bNS && !(bFirstStep && inputrec->bContinuation)) {
      bMasterState = FALSE;
      /* Correct the new box if it is too skewed */
      if (DYNAMIC_BOX(*inputrec) && !bRerunMD) {
	if (correct_box(state->box,graph))
	  bMasterState = TRUE;
      }
      if (DOMAINDECOMP(cr) && bMasterState)
	dd_collect_state(cr->dd,&top_global->blocks[ebCGS],state,state_global);
      
      if (DOMAINDECOMP(cr)) {
	/* Repartition the domain decomposition */
	wallcycle_start(wcycle,ewcDOMDEC);
	dd_partition_system(stdlog,step,cr,bMasterState,
			    state_global,top_global,inputrec,
			    state,&f,&buf,mdatoms,top,fr,vsite,constr,
			    nrnb,wcycle,do_verbose);
	wallcycle_stop(wcycle,ewcDOMDEC);
      }
    }

    if (MASTER(cr) && do_log && !bFFscan)
      print_ebin_header(log,step,t,state->lambda);

    /* Set values for invmass etc. This routine not parallellized, but hardly
     * ever used, only when doing free energy calculations.
     */
    if(inputrec->efep != efepNO)
      update_mdatoms(mdatoms,state->lambda); 
    
    clear_mat(force_vir);
    
    /* Ionize the atoms if necessary */
    if (bIonize)
      ionize(log,mdatoms,top->atoms.atomname,t,inputrec,state->x,state->v,
	     mdatoms->start,mdatoms->start+mdatoms->homenr,state->box,cr);
      
    /* Update force field in ffscan program */
    if (bFFscan) {
      if (update_forcefield(nfile,fnm,fr,mdatoms->nr,state->x,state->box)) {
	if (gmx_parallel_env)
	  gmx_finalize(cr);
	exit(0);
      }
    }

    GMX_MPE_LOG(ev_timestep2);

    if (bShell_FlexCon) {
      /* Now is the time to relax the shells */
      count=relax_shells(log,cr,bVerbose,bFFscan ? step+1 : step,
			 inputrec,bNS,bStopCM,top,constr,ener,fcd,
			 state,f,buf,mdatoms,
			 nrnb,wcycle,graph,grps,
			 nshell,shells,fr,t,mu_tot,
			 state->natoms,&bConverged,vsite,
			 fp_field);
      tcount+=count;
      
      if (bConverged)
	nconverged++;
    }
    else {
      /* The coordinates (x) are shifted (to get whole molecules) in do_force
       * This is parallellized as well, and does communication too. 
       * Check comments in sim_util.c
       */
      do_force(log,cr,inputrec,step,nrnb,wcycle,top,grps,
	       state->box,state->x,f,buf,mdatoms,ener,fcd,
	       state->lambda,graph,
	       TRUE,bNS,FALSE,TRUE,fr,mu_tot,FALSE,t,fp_field,edyn);
    }

    GMX_BARRIER(cr->mpi_comm_mygroup);

    if (bTCR)
      mu_aver = calc_mu_aver(cr,state->x,mdatoms->chargeA,
			     mu_tot,top,mdatoms,gnx,grpindex);
    if (bGlas)
      do_glas(log,mdatoms->start,mdatoms->homenr,state->x,f,
	      fr,mdatoms,top->idef.atnr,inputrec,ener);
    
    if (bTCR && bFirstStep) {
      tcr=init_coupling(log,nfile,fnm,cr,fr,mdatoms,&(top->idef));
      fprintf(log,"Done init_coupling\n"); 
      fflush(log);
    }

    /* Now we have the energies and forces corresponding to the 
     * coordinates at time t. We must output all of this before
     * the update.
     * for RerunMD t is read from input trajectory
     */
    if (vsite) {
      wallcycle_start(wcycle,ewcVSITESPREAD);
      spread_vsite_f(log,vsite,state->x,f,fr->fshift,nrnb,
		     &top->idef,fr->ePBC,fr->bMolPBC,graph,state->box,cr);
      wallcycle_stop(wcycle,ewcVSITESPREAD);
    }

    GMX_MPE_LOG(ev_virial_start);
    /* Calculation of the virial must be done after vsites!    */
    /* Question: Is it correct to do the PME forces after this? */
    calc_virial(log,mdatoms->start,mdatoms->homenr,state->x,f,
		force_vir,fr->vir_el_recip,graph,state->box,nrnb,fr);
    GMX_MPE_LOG(ev_virial_finish);

    /* Spread the LR force on virtual sites to the other particles... 
     * This is parallellized. MPI communication is performed
     * if the constructing atoms aren't local.
     */
    if (vsite && fr->bEwald) {
      wallcycle_start(wcycle,ewcVSITESPREAD);
      spread_vsite_f(log,vsite,state->x,fr->f_el_recip,NULL,nrnb,
		     &top->idef,fr->ePBC,fr->bMolPBC,graph,state->box,cr);
      wallcycle_stop(wcycle,ewcVSITESPREAD);
    }

    GMX_MPE_LOG(ev_sum_lrforces_start);
    sum_lrforces(f,fr,mdatoms->start,mdatoms->homenr);
    GMX_MPE_LOG(ev_sum_lrforces_finish);

    GMX_MPE_LOG(ev_output_start);

    bX   = (do_per_step(step,inputrec->nstxout) || bLastStep);
    bV   = (do_per_step(step,inputrec->nstvout) || bLastStep);
    bF   = (do_per_step(step,inputrec->nstfout));
    bXTC = (do_per_step(step,inputrec->nstxtcout));

    if (bX || bV || bF || bXTC) {
      wallcycle_start(wcycle,ewcTRAJ);
      write_traj(cr,fp_trn,bX,bV,bF,fp_xtc,bXTC,inputrec->xtcprec,
		 top_global,step,t,state,state_global,f,f_global);
      debug_gmx();

      if (bLastStep && (Flags & MD_CONFOUT) && MASTER(cr) &&
	  !bRerunMD && !bFFscan) {
	/* x and v have been collected in write_traj */
	fprintf(stderr,"Writing final coordinates.\n");
	write_sto_conf(ftp2fn(efSTO,nfile,fnm),
		       *top_global->name,&top_global->atoms,
		       state_global->x,state_global->v,state->box);
	debug_gmx();
      }
      wallcycle_stop(wcycle,ewcTRAJ);
    }
    GMX_MPE_LOG(ev_output_finish);

    clear_mat(shake_vir);
    
    /* Afm and Umbrella type pulling happens before the update, 
     * other types in update 
     */
    if (inputrec->ePull == epullUMBRELLA || inputrec->ePull == epullCONST_F) {
      ener[F_COM_PULL] =
	pull_potential(inputrec->ePull,inputrec->pull,
		       state->x,f,force_vir,state->box,
		       top,inputrec->init_t+step*inputrec->delta_t,mdatoms,cr);
      /* Avoid double counting */
      if (!MASTER(cr))
	ener[F_COM_PULL] = 0;
    }

    if (bFFscan)
      clear_rvecs(state->natoms,buf);

    /* Box is changed in update() when we do pressure coupling,
     * but we should still use the old box for energy corrections and when
     * writing it to the energy file, so it matches the trajectory files for
     * the same timestep above. Make a copy in a separate array.
     */
    copy_mat(state->box,lastbox);
 
    
    GMX_MPE_LOG(ev_update_start);
    /* This is also parallellized, but check code in update.c */
    /* bOK = update(nsb->natoms,START(nsb),HOMENR(nsb),step,state->lambda,&ener[F_DVDL], */
    bOK = TRUE;
    if (!bRerunMD || rerun_fr.bV || bForceUpdate) {
      wallcycle_start(wcycle,ewcUPDATE);
      update(step,&ener[F_DVDL],inputrec,mdatoms,state,graph,f,buf,
	     top,grps,shake_vir,cr,nrnb,sd,constr,edyn,bHaveConstr,bNEMD,
	     TRUE,bFirstStep,pres);
      wallcycle_stop(wcycle,ewcUPDATE);
    } else {
      /* Need to unshift here */
      if ((inputrec->ePBC == epbcXYZ) && (graph->nnodes > 0))
	unshift_self(graph,state->box,state->x);
    }

    GMX_BARRIER(cr->mpi_comm_mygroup);
    GMX_MPE_LOG(ev_update_finish);

    if (!bOK && !bFFscan)
      gmx_fatal(FARGS,"Constraint error: Shake, Lincs or Settle could not solve the constrains");

    /* Non-equilibrium MD: this is parallellized, but only does communication
     * when there really is NEMD.
     */
    if (PAR(cr) && bNEMD) 
      accumulate_u(cr,&(inputrec->opts),grps);
      
    debug_gmx();
    if (grps->cosacc.cos_accel == 0)
      calc_ke_part(state->v,&(inputrec->opts),mdatoms,grps,nrnb,state->lambda);
    else
      calc_ke_part_visc(state->box,state->x,state->v,&(inputrec->opts),
			mdatoms,grps,nrnb,state->lambda);

    /* since we use the new coordinates in calc_ke_part_visc, we should use
     * the new box too. Still, won't this be offset by one timestep in the
     * energy file? / EL 20040121
     */ 

    debug_gmx();
    /* Calculate center of mass velocity if necessary, also parallellized */
    if (bStopCM && !bFFscan && !bRerunMD)
      calc_vcm_grp(log,mdatoms->start,mdatoms->homenr,mdatoms,
		   state->x,state->v,vcm);

    /* Check whether everything is still allright */    
    if (bGotTermSignal || bGotUsr1Signal) {
      if (bGotTermSignal)
	terminate = 1;
      else
	terminate = -1;
      fprintf(log,"\n\nReceived the %s signal\n\n",
	      bGotTermSignal ? "TERM" : "USR1");
      fflush(log);
      if (MASTER(cr)) {
	fprintf(stderr,"\n\nReceived the %s signal\n\n",
	      bGotTermSignal ? "TERM" : "USR1");
	fflush(stderr);
      }
      bGotTermSignal = FALSE;
      bGotUsr1Signal = FALSE;
    }

    if (PAR(cr)) {
      wallcycle_start(wcycle,ewcMoveE);
      /* Globally (over all NODEs) sum energy, virial etc. 
       * This includes communication 
       */
      global_stat(log,cr,ener,force_vir,shake_vir,mu_tot,
		  inputrec,grps,constr,vcm,&terminate);

      /* Correct for double counting energies, should be moved to 
       * global_stat 
       */
      if (fr->bTwinRange && !bNS) 
	for(i=0; (i<grps->estat.nn); i++) {
	  grps->estat.ee[egCOULLR][i] /= (cr->nnodes-cr->npmenodes);
	  grps->estat.ee[egLJLR][i]   /= (cr->nnodes-cr->npmenodes);
	}
      wallcycle_stop(wcycle,ewcMoveE);
    }
      
    /* This is just for testing. Nothing is actually done to Ekin
     * since that would require extra communication.
     */
    if (!bNEMD && debug && (vcm->nr > 0))
      correct_ekin(debug,mdatoms->start,mdatoms->start+mdatoms->homenr,
		   state->v,vcm->group_p[0],
		   mdatoms->massT,mdatoms->tmass,ekin);
    
    if ((terminate != 0) && (step - inputrec->init_step < inputrec->nsteps)) {
      if (terminate<0 && inputrec->nstxout)
	/* this is the USR1 signal and we are writing x to trr, 
	   stop at next x frame in trr */
	inputrec->nsteps =
	  (step/inputrec->nstxout + 1) * inputrec->nstxout - inputrec->init_step;
      else
	inputrec->nsteps = step + 1 - inputrec->init_step;
      fprintf(log,"\nSetting nsteps to %d, last step is %d\n\n",
	      inputrec->nsteps,inputrec->init_step+inputrec->nsteps);
      fflush(log);
      if (MASTER(cr)) {
	fprintf(stderr,"\nSetting nsteps to %d, last step is %d\n\n",
		inputrec->nsteps,inputrec->init_step+inputrec->nsteps);
	fflush(stderr);
      }
      /* erase the terminate signal */
      terminate = 0;
    }
      
     /* Do center of mass motion removal */
    if (bStopCM && !bFFscan && !bRerunMD) {
      check_cm_grp(log,vcm,1);
      do_stopcm_grp(log,mdatoms->start,mdatoms->homenr,mdatoms->cVCM,
		    state->x,state->v,vcm);
      inc_nrnb(nrnb,eNR_STOPCM,mdatoms->homenr);
      /*
      calc_vcm_grp(log,START(nsb),HOMENR(nsb),mdatoms->massT,x,v,vcm);
      check_cm_grp(log,vcm);
      do_stopcm_grp(log,START(nsb),HOMENR(nsb),x,v,vcm);
      check_cm_grp(log,vcm);
      */
    }
    
    /* Add force and shake contribution to the virial */
    m_add(force_vir,shake_vir,total_vir);

    /* Sum the potential energy terms from group contributions */
    sum_epot(&(inputrec->opts),grps,ener);

    /* Calculate the amplitude of the cosine velocity profile */
    grps->cosacc.vcos = grps->cosacc.mvcos/mdatoms->tmass;

    /* Sum the kinetic energies of the groups & calc temp */
    ener[F_TEMP] = sum_ekin(bRerunMD,&(inputrec->opts),grps,ekin,
			    &(ener[F_DVDLKIN]));
    ener[F_EKIN] = trace(ekin);

    /* Calculate Temperature coupling parameters lambda and adjust
     * target temp when doing simulated annealing
     */
    /*
    if(inputrec->etc==etcBERENDSEN)
      berendsen_tcoupl(&(inputrec->opts),grps,inputrec->delta_t);
    else if(inputrec->etc==etcNOSEHOOVER)
      nosehoover_tcoupl(&(inputrec->opts),grps,inputrec->delta_t);
    */

    /* Calculate pressure and apply LR correction if PPPM is used.
     * Use the box from last timestep since we already called update().
     */
    calc_pres(fr->ePBC,inputrec->nwall,lastbox,ekin,total_vir,pres,
	      (fr->eeltype==eelPPPM) ? ener[F_COUL_RECIP] : 0.0);
    
    /* Calculate long range corrections to pressure and energy */
    if (bTCR || bFFscan)
      set_avcsixtwelve(log,fr,&top_global->atoms,&top_global->blocks[ebEXCLS]);
      
    /* Calculate long range corrections to pressure and energy */
    calc_dispcorr(log,inputrec,fr,step,top_global->atoms.nr,
		  lastbox,state->lambda,
		  pres,total_vir,ener);

    ener[F_ETOT] = ener[F_EPOT] + ener[F_EKIN];

    if (inputrec->etc == etcNOSEHOOVER)
      ener[F_ECONSERVED] =
	ener[F_ETOT] + nosehoover_energy(&(inputrec->opts),grps,
					 state->nosehoover_xi);

    /* Check for excessively large energies */
    if (bIonize) {
#ifdef GMX_DOUBLE
      real etot_max = 1e200;
#else
      real etot_max = 1e30;
#endif
      if (fabs(ener[F_ETOT]) > etot_max) {
	fprintf(stderr,"Energy too large (%g), giving up\n",ener[F_ETOT]);
	break;
      }
    }

    /* The coordinates (x) were unshifted in update */
    if (bFFscan && (!bShell_FlexCon || bConverged))
      if (print_forcefield(log,ener,mdatoms->homenr,f,buf,xcopy,
			   &(top->blocks[ebMOLS]),mdatoms->massT,pres)) {
	if (gmx_parallel_env)
	  gmx_finalize(cr);
	fprintf(stderr,"\n");
	exit(0);
      }
    
    if (bTCR) {
      /* Only do GCT when the relaxation of shells (minimization) has converged,
       * otherwise we might be coupling to bogus energies. 
       * In parallel we must always do this, because the other sims might
       * update the FF.
       */
      
      /* Since this is called with the new coordinates state->x, I assume
       * we want the new box state->box too. / EL 20040121
       */
      do_coupling(log,nfile,fnm,tcr,t,step,ener,fr,
		  inputrec,MASTER(cr),
		  mdatoms,&(top->idef),mu_aver,
		  top->blocks[ebMOLS].nr,cr,
		  state->box,total_vir,pres,
		  mu_tot,state->x,f,bConverged);
      debug_gmx();
    }

    /* Time for performance */
    if (((step % stepout) == 0) || bLastStep)
      update_time();

    /* Output stuff */
    if (MASTER(cr)) {
      bool do_ene,do_dr,do_or,do_dihr;
      
      upd_mdebin(mdebin,fp_dgdl,mdatoms->tmass,step_rel,t,ener,state,lastbox,
		 shake_vir,force_vir,total_vir,pres,grps,mu_tot,constr);
      do_ene = do_per_step(step,inputrec->nstenergy) || bLastStep;
      do_dr  = do_per_step(step,inputrec->nstdisreout) || bLastStep;
      do_or  = do_per_step(step,inputrec->nstorireout) || bLastStep;
      do_dihr= do_per_step(step,inputrec->nstdihreout) || bLastStep;
      print_ebin(fp_ene,do_ene,do_dr,do_or,do_dihr,do_log?log:NULL,
		 step,step_rel,t,
		 eprNORMAL,bCompact,mdebin,fcd,&(top->atoms),&(inputrec->opts));
      if (inputrec->ePull != epullNO)
	pull_print_output(inputrec->pull,step,t);

      if (bVerbose)
	fflush(log);
    }
    
    /* Remaining runtime */
    if (MULTIMASTER(cr) && do_verbose) {
      if (bShell_FlexCon)
	fprintf(stderr,"\n");
      print_time(stderr,start_t,step,inputrec);
    }

    /* Replica exchange */
    bExchanged = FALSE;
    if ((repl_ex_nst > 0) && (step > 0) && !bLastStep &&
	do_per_step(step,repl_ex_nst))
      bExchanged = replica_exchange(log,cr,repl_ex,state_global,ener[F_EPOT],
				    &(top_global->blocks[ebCGS]),state,
				    step,t);
    if (bExchanged && PAR(cr)) {
      if (DOMAINDECOMP(cr))
	dd_partition_system(stdlog,step,cr,TRUE,
			    state_global,top_global,inputrec,
			    state,&f,&buf,mdatoms,top,fr,vsite,constr,
			    nrnb,wcycle,FALSE);
      else
	pd_distribute_state(cr,state);
    }
    
    bFirstStep = FALSE;
    
    if (bRerunMD) 
      /* read next frame from input trajectory */
      bNotLastFrame = read_next_frame(status,&rerun_fr);

    if (!bRerunMD || !rerun_fr.bStep) {
      /* increase the MD step number */
      step++;
      step_rel++;
    }
  }
  /* End of main MD loop */
  debug_gmx();

  if (bRerunMD)
    close_trj(status);
	  
  if (MASTER(cr)) {
    print_ebin(fp_ene,FALSE,FALSE,FALSE,FALSE,log,step,step_rel,t,
	       eprAVER,FALSE,mdebin,fcd,&(top->atoms),&(inputrec->opts));
    print_ebin(fp_ene,FALSE,FALSE,FALSE,FALSE,log,step,step_rel,t,
	       eprRMS,FALSE,mdebin,fcd,&(top->atoms),&(inputrec->opts));
    close_enx(fp_ene);
    if (inputrec->nstxtcout)
      close_xtc(fp_xtc);
    close_trn(fp_trn);
    if (fp_dgdl)
      fclose(fp_dgdl);
    if (fp_field)
      fclose(fp_field);
  }
  debug_gmx();
  
  /* clean up edsam stuff, no effect if edyn->bEdsam == FALSE */
  finish_edsam(stdlog,top,inputrec,mdatoms,mdatoms->start,mdatoms->homenr,cr,
	       edyn);

  if (bShell_FlexCon) {
    fprintf(log,"Fraction of iterations that converged:           %.2f %%\n",
	    (nconverged*100.0)/step_rel);
    fprintf(log,"Average number of force evaluations per MD step: %.2f\n",
	    tcount/step_rel);
  }

  if (repl_ex_nst > 0 && MASTER(cr))
    print_replica_exchange_statistics(log,repl_ex);
    
  return start_t;
}
