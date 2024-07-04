## README ##

`PICASA` : `P`arameter `I`nference using `C`obaya with `A`nisotropic `S`imulated `A`nnealing

Application of anisotropic simulated annealing (ASA) to augment MCMC parameter inference for expensive likelihoods as described in [arXiv:2205.07906](https://arxiv.org/abs/2205.07906).

### Requirements ###

Python3.6 or higher with [NumPy](https://numpy.org) and [SciPy](https://www.scipy.org) (for basic ASA usage), [Cobaya](https://cobaya.readthedocs.io), [Scikit-Learn](https://scikit-learn.org) and [GetDist](https://getdist.readthedocs.io)  (for interfacing ASA with MCMC inference) and optionally [Matplotlib](https://matplotlib.org) and [Jupyter](https://jupyter.org) (for example usage).

### Description ###

The folder **code/** contains the following scripts:

1. **asa_utilities.py** : This has the single class `ASAUtilities` containing all necessary functionality for implementing the ASA algorithm on generic likelihood calculations in D-dimensional parameter space.

2. **asa_likelihood.py** : This has the class `ASALike` which implements the interface between ASA and MCMC sampling using Cobaya.

3. **picasa.py**: This has the class `PICASA` containing black-box implementations of ASA (method `optimize`) and ASA + MCMC (method `mcmc`).

The folder also contains a .yaml file specifying the defaults of ASALike(), along with a default data set, and a sub-folder **example_likelihoods/** containing likelihoods used in the examples.

The user supplies a dictionary containing (among other things) Python callables to evaluate the cost function (interpreted as minus 1/2 times the likelihood) for a specified data set given a particular model. (See the code snippet below as well as the script **code/demo.py** and especially the docstring of `PICASA.mcmc()` for a detailed description of the keys expected in the dictionary.) The ASA algorithm minimises the cost function and provides a sampling of the likelihood on a sequence of Latin hypercubes. The interface to MCMC then constructs an interpolating function for the D-dimensional cost function (using Gaussian Process Regression by default, implemented using Scikit-Learn) and passes this to Cobaya which performs a standard MCMC search. 

### Examples ###

The Jupyter notebook **examples/TestPICASA.ipynb** contains example usage for some linear and non-linear problems, comparing the performance and output of PICASA with that of a standard MCMC using Cobaya (needs Cobaya, Scikit-Learn, Matplotlib and GetDist). 

The code snippet below demonstrates a simple example (see **code/demo.py** for more detailed output generation)

### 
	import numpy as np
	import os
	from picasa import PICASA
	import matplotlib.pyplot as plt
	from getdist.mcsamples import loadMCSamples
	import getdist.plots as gdplt

	class MyStuff(object):
	    """ This class contains custom model and cost function definitions.
		 These could be defined directly as functions, or could be in separate classes, imported from elsewhere.
	    """

	    def __init__(self,rng=None):
		    self.rng = rng if rng is not None else np.random.RandomState(seed=42)
		    # Data structure (observables and control variables, here y = f(x)). Could also be loaded from file.
		    # ... x values
		    self.xvals = np.linspace(-1,1,10)
		    # ... inverse covariance matrix
		    err = 0.2
		    self.invcov_data = (1/err**2)*np.eye(self.xvals.size)
		    # ... y values: effectively, f(x) := x
		    self.data = self.xvals + err*self.rng.randn(self.xvals.size)


	    def my_model(self,x,params):
		    return np.sum(np.array([params[i]*x**i for i in range(len(params))]),axis=0)

	    def my_cost(self,data,invcov_data,model,cost_args):
		    residual = data - model
		    chi2 = np.dot(residual,np.dot(invcov_data,residual))
		    return chi2


	rng = np.random.RandomState(seed=42)
	pic = PICASA(base_dir='../') # base_dir should be the top-level folder on local system.
	# pic.NPROC = 6 # uncomment this to set no. of processes by hand
	stuff = MyStuff(rng=rng)

	# let's fit a quadratic polynomial to the data
	params_true = [0.,1.,0.] # true values, for comparison
	degree = 2
	dim = degree + 1

	# analytical answer
	err_data = 1/np.sqrt(np.diag(stuff.invcov_data))
	apoly,cov_poly = pic.polyfit_custom(stuff.xvals,stuff.data,degree,sig2=err_data)
	sig_apoly = np.sqrt(np.diag(cov_poly))
	
	# PICASA 
	# see script demo.py for details regarding mandatory and optional keys
    data_pack = {}
    data_pack['chi2_file'] = 'examples/demo_chi2data.txt'
    data_pack['Nsamp'] = 10 # number of samples at each zoom iteration
    data_pack['param_mins'] = [-4.0]*dim # starting range for 
    data_pack['param_maxs'] = [4.0]*dim # parameter exploration
    data_pack['dsig'] = 5.0 # maximum factor multiplying 1-sigma along each parameter direction for last iterations
    data_pack['data'] = stuff.data
    data_pack['invcov_data'] = stuff.invcov_data
    data_pack['cost_func'] = stuff.my_cost
    data_pack['cost_args'] = None
    data_pack['model_func'] = stuff.my_model
    data_pack['model_args'] = stuff.xvals
    data_pack['rng'] = rng # Default None
    data_pack['parallel'] = True
    data_pack['chain_dir'] = 'examples/chains/demo'
    data_pack['mcmc_prior'] = [[-0.3,0.5],[0.5,1.5],[-0.7,0.7]] # Default None

    start_time = time()
    info,sampler,chi2,params,cov_asa,params_best,chi2_min = pic.mcmc(data_pack)
    pic.time_this(start_time)

    gd_sample = loadMCSamples(os.path.abspath(info["output"]))
    # samples contain a0..a(dim-1) chi2 | chi2__name
    ibest = gd_sample.samples.T[dim].argmin()
    mcmc_best = gd_sample.samples[ibest,:dim]	

    print('\n Residuals (percentage) relative to analytical answer...')
    print("... Evaluated best-fit ( a0,..,a{0:d} ) = ( ".format(dim-1)+','.join(['%.1e' % (pval,) for pval in np.fabs(params[chi2.argmin()]/apoly - 1.0)*1e2])+" )")
    print("... Quadratic form ( a0,..,a{0:d} ) = ( ".format(dim-1)+','.join(['%.1e' % (pval,) for pval in np.fabs(params_best/apoly - 1.0)*1e2])+" )")
    print("... MCMC (final) ( a0,..,a{0:d} ) = ( ".format(dim-1)+','.join(['%.1e' % (pval,) for pval in np.fabs(mcmc_best/apoly - 1.0)*1e2])+" )")
    print('\nTotal number of function evaluations = {0:d}'.format(chi2.size))

    # Plots
    gdplot = gdplt.get_subplot_plotter(subplot_size=2.)
    plot_param_list = ['a'+str(d) for d in range(dim)]
    plot_limits_dict = {}
    for d in range(dim):
	    plot_limits_dict['a'+str(d)] = (info['params']['a'+str(d)]['prior']['min'],
					info['params']['a'+str(d)]['prior']['max'])

    gdplot.triangle_plot([gd_sample], plot_param_list, 
			 filled=True,contour_colors=['crimson'],legend_loc='upper center',
			 param_limits=plot_limits_dict)
    gdplot.add_text('$N_{{\\rm eval,PICASA}} = {0:d}$'.format(chi2.size),
		    x=-0.75,y=2.35,fontsize=14,c='crimson')
    for par_x in range(dim):
	    for par_y in range(par_x,dim):
		if par_x != par_y:
			    ax = gdplot.get_axes_for_params('a'+str(par_x),'a'+str(par_y))
			    ax.axhline(params_true[par_y],c='k',ls=':',lw=1)
			    ax.axhline(params_best[par_y],c='crimson',ls='-',lw=1.,alpha=0.6)
			    ax.axhline(apoly[par_y],c='b',ls='--',lw=1.,alpha=0.6)

			    ax.axvline(params_true[par_x],c='k',ls=':',lw=1)
			    ax.axvline(params_best[par_x],c='crimson',ls='-',lw=1.,alpha=0.6)
			    ax.axvline(apoly[par_x],c='b',ls='--',lw=1.,alpha=0.6)
		else:
			    ax = gdplot.subplots[par_x,par_x]
			    ax.axvline(params_true[par_x],c='k',ls=':',lw=1)
			    ax.axvline(params_best[par_x],c='crimson',ls='-',lw=1.,alpha=0.6)
			    ax.axvline(apoly[par_x],c='b',ls='--',lw=1.,alpha=0.6)
    gdplot.export(fname='demo_results.pdf',adir='../examples/plots/')

    Nsample = gd_sample.samples.shape[0]
    Nboot = 500
    ind = rng.choice(Nsample,size=np.min([Nboot,Nsample]),replace=False)
    model_b = np.zeros((Nboot,stuff.xvals.size),dtype=float)
    for b in range(Nboot):
	model_b[b] = stuff.my_model(stuff.xvals,gd_sample.samples[ind[b],:dim])

    picasa_16pc = np.percentile(model_b,16,axis=0)
    picasa_84pc = np.percentile(model_b,84,axis=0)

    plt.figure(figsize=(5,5))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(stuff.xvals,stuff.my_model(stuff.xvals,params_true),'k:',lw=2,label='true')
    plt.errorbar(stuff.xvals,stuff.data,yerr=err_data,capsize=5,elinewidth=0.5,c='purple',ls='none',marker='o')
    plt.plot(stuff.xvals,stuff.my_model(stuff.xvals,params_best),'-',c='crimson',lw=2,label='PICASA')
    plt.fill_between(stuff.xvals,picasa_16pc,picasa_84pc,color='crimson',alpha=0.2)
    plt.plot(stuff.xvals,stuff.my_model(stuff.xvals,apoly),'--',c='b',lw=0.8,label='analytical')
    plt.legend(loc='upper left')
    plt.savefig('../examples/plots/demo_data.pdf',bbox_inches='tight')

![example: output](examples/plots/demo_results.pdf)
![example: data](examples/plots/demo_data.pdf)

### Credit ###

Paranjape (2022), [arXiv:2205.07906](https://arxiv.org/abs/2205.07906)

### Contact ###

Aseem Paranjape
