#imports
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import plotly.express as ex
from finufft import nufft1d3
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from scipy.stats import trim_mean, chi2
from math import ceil
import seaborn as sn

plt.rcParams.update({"font.size" : 14})
plt.rcParams.update({"axes.labelsize" : 14})

#Class definition
class fdac:
    
    '''
    Class containing the frequency domain object and performs activity correction 
    
    :var t: list of observations times 
    :type t: list
    :var first_com: List with values of the main variable which the linear regression will be perfomed to (eg. RV).
    :type first_com: list
    :var rest_com: Nested list of explanatory variables used as the principal observation that develop the linear regression model.
    :type rest_com: list
    :var names: Names used to identify the first component and the further components the same order as :var:`firs_com` and :var:`rest_com`.
    :type names: str list
    '''
    
    def __init__(self, t, first_com, rest_com, names = None, show_freq = True):
        try:
            test1 = (len(t) == len(first_com))
            test2 = (len(t) == len(rest_com[0]))
            if (not test1) or (not test2):
                raise ValueError
        except ValueError:
            print("Number of observation times must be equal to the number of values in the first and further components")
            print("No object created")
            return
        ind = np.argsort(t) # sort by time
        self.t = np.array(t[ind])
        if (self.t[0] != 0.0): # shift so time series starts at t=0
            self.t = self.t - self.t[0] 
        self.N = len(self.t)
        self.dts = np.diff(self.t)
        #Select only the arrays that have nan values in the rest_com array 
        mask = np.any(np.isnan(rest_com), axis=1)
        arrays_with_nan = rest_com[mask]
        if arrays_with_nan.shape[0] ==0:
            self.rest_nans= None
            self.t_nans=None
        if arrays_with_nan.shape[0] ==1:
            indices= ~np.isnan(arrays_with_nan[0])
            self.rest_nans=np.array(arrays_with_nan[0])[indices]
            self.rest_nans = self.rest_nans.astype(complex)
            self.t_nans =self.t[indices]
        else :
            t_rest_list =[]
            rest_list = []
            for i in range(arrays_with_nan.shape[0]):
                indices = ~np.isnan(arrays_with_nan[i])
                rest = np.array(arrays_with_nan[i])[indices]
                t_rest= self.t[indices]
                rest_list.append(rest)
                t_rest_list.append(t_rest)
                
            self.t_nans = np.array(t_rest_list, dtype=object)
            self.rest_nans = np.array(rest_list, dtype=object)
        
        self.first_com = np.array(first_com).astype(complex)
        self.rest_com = np.array(rest_com[~mask]).astype(complex)
        if names != None:
            try:
                test3 = len(names) == 1 + len(rest_com)
                if (not test3):
                    raise ValueError
            except ValueError:
                print("Lenght of the names does not match the number of variables")
                print('The name variable has been set to None')
                self.names = None
            self.names = names
        
        # various possibilities for the Nyquist frequency; Rayleigh resolution
        self.Nyq_meandt = 0.5/np.mean(self.dts)
        self.Nyq_tmeandt_10 = 0.5/trim_mean(self.dts, 0.05)
        self.Nyq_tmeandt_20 = 0.5/trim_mean(self.dts, 0.1)
        self.Nyq_meddt = 0.5/np.median(self.dts)

        #Nyquist frequencies for the
        
        if show_freq:
            print('Mean Nyquist frequency: ', self.Nyq_meandt,'\n')
            print('Median Nyquist frequency: ', self.Nyq_meddt, '\n')
            print('10% trimmed Mean Nyquist frequency: ', self.Nyq_tmeandt_10, '\n')
            print('20% trimmed mean Nyqusit frequenct: ', self.Nyq_tmeandt_20)
        
        #values not computed yet used to raise errors if the functions are not run in order
        self.fgrid = None
        self.FFTframe = None
        self.FFT_real_ac = None
        self.FFT_imag_ac = None
        self.pc = None
        self.activity = None
        
    def plot_corr_matrix(self):
        '''
        Generates a matrix plot of the correlation between the explanatory variables, using Pearson's method.
        '''
        plt.figure(figsize = (9,5))
        act_dfram = pd.DataFrame((self.rest_com).T) #transpose the data so that it works properly
        if self.names == None:
            sn.heatmap(act_dfram.corr())
        else:
            sn.heatmap(act_dfram.corr(), xticklabels = self.names[1:], yticklabels = self.names[1:])
        
    def frequency_grid(self, Nyquist, oversample=4): 
        # "Nyquist" frequency is defined by user. It can be one of the four possibilities 
        #     computed when the time series object was constructed, or something else entirely.
        # With the oversample keyword, the user can go above (or below) the Rayleigh resolution:
        #     the frequency grid spacing is self.Rayleigh / oversample
        try:
            if Nyquist < 0:
                raise ValueError
        except ValueError:
            print("Nyquist frequency must be float > 0 - no frequency grid calculated")
            return
        try:
            good_oversample = (((type(oversample) is int) or (type(oversample) is float)) \
                              and (oversample > 0))
            if not good_oversample:
                raise ValueError
        except ValueError:
            print("Oversample must be number > 0 - no frequency grid calculated")
            return
        self.Rayleigh = 1/ np.max(self.t)
        df = self.Rayleigh / oversample
        self.nf = int(Nyquist // df) # Number of POSITIVE (non-zero) frequencies in the grid
        if (self.nf % 2) != 0: # make sure zero frequency is included
            self.nf += 1
        if (Nyquist > self.Nyq_meddt):
            print("Warning: your requested Nyquist frequency is higher than the Nyquist")
            print("frequency associated with the median timestep. Make sure that makes")
            print("sense for your dataset.")
        self.fgrid = np.linspace(-Nyquist, Nyquist, num=2*self.nf+1, endpoint=True)
        # print(self.fgrid)
        #self.powfgrid = self.fgrid[self.nf:]
    '''
    def frequency_grid(self, Nyquistf):
        

        Computes the frequency grid given a specific type of Nyquist frequency calculated for the time series
        
        :param Nyquistf: Nyquist frequency used to compute the frequency domain. Usually uses one of the frequencies calculated when initializing :class:`fdac`
        :type Nyquistf: float

        try:
            if Nyquistf < 0:
                raise ValueError
        except ValueError:
            print("Nyquist frequency must be float > 0 - no frequency grid calculated")
            return
        
        self.maxf = 2* Nyquistf      # Maximum frequency is 2x the Nyquist frequency
        self.Ray = 1/ np.max(self.t)
        self.nR_pos = ceil(self.maxf / self.Ray) #Number of Rayleigh resolutions
        #self.nR_pos = int(self.maxf / self.Ray) + 1 #Number of Rayleigh resolutions
        print("Number of Rayleigh resolution units:", self.nR_pos)
        self.nf = 2*self.nR_pos + 1     
        self.fgrid = np.linspace(-self.maxf, self.maxf, num= self.nf, endpoint=True)
        print("Maximum frequency (cycles/day):", f"{self.maxf:.4f}")
        print("Rayleigh resolution:", f"{self.Ray:.4f}")
    '''
        
    #Computes the NFFT of the activity indicators   
    def computeNFFT(self):
        
        '''
        Computes the Non-Uniform Fast Fourier Transform (NFFT) for each observation
        '''
        try:
            not_gridded = (self.fgrid is None)
            if not_gridded:
                raise ValueError
        except ValueError:
            print("You must call frequency_grid() before calling pow_FT()")
            print("No Fourier transform or power spectrum calculated")
            return
        
        #Compute the NFFT for the arrays without nans
        self.obs = np.vstack([self.first_com, self.rest_com])
        self.fft_list = []
        for i in range(len(self.obs)):
            fft_val = nufft1d3(2*np.pi*self.t, self.obs[i] - self.obs[i].mean(), \
                  self.fgrid, isign=-1, nthreads=1)
            self.fft_list.append(fft_val)
        #Compute the NFFT if there are arrays that have nans
        if self.t_nans!= None:
            if self.rest_nans.ndim ==1:
                fft_val = nufft1d3(2*np.pi*self.t_nans, self.rest_nans - self.rest_nans.mean(), \
                    self.fgrid, isign=-1, nthreads=1)
                self.fft_list.append(fft_val)
            else :
                for i in range(len(self.rest_nans)):
                    fft_val = nufft1d3(2*np.pi*self.t_nans[i], self.rest_nans[i] - self.rest_nans[i].mean(), \
                                    self.fgrid, isign=-1, nthreads=1)
                    self.fft_list.append(fft_val)
            
        #separate the FFT of the first component from the rest of the components
        self.first_FFT = self.fft_list[0]
        self.rest_FFT = self.fft_list[1:]
        
        # data frame with columns = real + imaginary parts of FFTs
        self.FFTframe = pd.DataFrame()
        self.FFTframe['frequency'] = self.fgrid
        
        #pack the FFT of the all the components
        for i in range(1,len(self.fft_list)):
            cname = self.names[i]
            self.FFTframe[cname+" re"] = self.fft_list[i].real 
            self.FFTframe[cname+" im"] = self.fft_list[i].imag #add abs value to this part and the computer will just interpret it at

        self.scaledFFT = StandardScaler().fit_transform(self.FFTframe.iloc[:,1:])
        ## Comment: I am not sure if this needs to be updated to also scale the ffts of the RVs to unit variance and no mean.
    
    def plotNFFT(self):
        
        '''
        Plots the NFFT of each observation
        '''
        try:
            no_fft = (self.FFTframe is None)
            if no_fft:
                raise ValueError
        except ValueError:
                print("NFFT haven't been computed yet, compute NFFT of the variables to display their plot")
                print("No plot computed")
                return
        
        if len(self.obs)%2 ==0:
            fig, axarr = plt.subplots(nrows=ceil(len(self.obs)/2), ncols=2, sharex=True, figsize=(12,9))
            for i, ax in enumerate(axarr.flat):
                ax.plot(self.fgrid, self.fft_list[i].real, label="Re", lw=0.8, color = 'b')
                ax.plot(self.fgrid, self.fft_list[i].imag, label="Im", lw=0.8, alpha=0.8, color = 'r')
                ax.set_title(self.names[i], fontsize="medium")
                ax.tick_params(labelsize="small")
                if (i % 2 == 0):
                    ax.set_ylabel(r"$\mathcal{F}\{x_t\}$")
                if (i == 0):
                    ax.legend(loc="best", fontsize="small")
                if (i >= (len(self.obs)-2)):
                    ax.set_xlabel("Frequency (cycles / day)")
        else:
            rows= ceil(np.sqrt(len(self.obs)))
            cols = ceil(len(self.obs)/rows)
            fig, axarr = plt.subplots(nrows=rows, ncols=cols, sharex=True, figsize=(12,9))
            # Flatten axs to easily access each subplot in a loop
            axarr = axarr.flatten()
            for i in range(len(self.obs)):
                axarr[i].plot(self.fgrid, self.fft_list[i].real, label="Re", lw=0.8, color = 'b')
                axarr[i].plot(self.fgrid, self.fft_list[i].imag, label="Im", lw=0.8, alpha=0.8, color = 'r')
                axarr[i].set_title(self.names[i], fontsize="medium")
                axarr[i].tick_params(labelsize="small")
                if (i % 2 == 0):
                    axarr[i].set_ylabel(r"$\mathcal{F}\{x_t\}$")
                if (i == 0):
                    axarr[i].legend(loc="best", fontsize="small")
                if ((i == len(self.obs)-2) or (i== (len(self.obs)-1))):
                    axarr[i].set_xlabel("Frequency (cycles / day)")
            # Hide any empty subplots
            for i in range(len(self.obs), len(axarr)):
                axarr[i].axis('off')  # Turn off the axis for extra subplots
    
    def pc_analysis(self, disp_variance = True):
        
        '''
        Performs the principal component analysis (PCA) of the observations excluding the main observatory variable
        :param disp_variance: plots the tota varaince as a function of components
        :type disp_variance: bool, optional
        '''
        
        ### This is not used for now so I will leave to improve for later###
        pc_an = PCA()
        self.pc = pc_an.fit_transform(self.scaledFFT)
        self.FFT_pc = pd.DataFrame(data=self.pc)
        self.FFT_pc.head()
        self.pc_loadings = pc_an.components_.T * \
               np.sqrt(pc_an.explained_variance_)
        explanatory = self.FFTframe.columns.tolist()[1:]
        self.loading_pc = pd.DataFrame(pc_loadings)
        self.loading_pc["variable"] = explanatory
        
        if disp_variacen ==  True:
            plt.figure(figsize=(9,5))
            plt.plot(np.cumsum(pc_analysis.explained_variance_ratio_), color = 'RoyalBlue')
            plt.xlabel('Number of components')
            plt.ylabel('Cumulative explained variance')
        
    def displayloadings(self):
        
        '''
        Display the loadings from the PCA analysis
        '''
        
        fig = ex.scatter(x=self.loading_pc.iloc[:,0], y=self.loading_pc.iloc[:,1], \
                 text=self.loading_pc['variable'],)
        fig.update_layout(height=500, width=500, title_text='Loadings', \
                         xaxis_title="Component 1", yaxis_title="Component 2",)
        fig.update_xaxes(range=[-1.1, 1.1])
        fig.update_yaxes(range=[-1.1, 1.1])
        fig.update_traces(textposition='bottom center')
        fig.add_shape(type="line", x0=-0, y0=-1.1, x1=-0, y1=1.1, line=dict(color="RoyalBlue",width=3))
        fig.add_shape(type="line", x0=-1.1, y0=0, x1=1.1, y1=0, line=dict(color="RoyalBlue",width=3))
        fig.show()
    
    def linear_reg(self, pca = False):
        
        '''
        Performs the linear regression to the NFFT of the main observatory variable using the rescaled value of the NFFT of the explanatory variables
        :param pca: Statement that decides if compute the linear regression using the PCA results. Default is False
        :type pca: bool, optional
        '''
        try:
            no_fft = (self.FFTframe is None)
            if no_fft:
                raise ValueError
        except ValueError:
            print("NFFT hasn't been computed yet, compute NFFT before proceeding")
            return
            
        if pca == True:
            try:
                no_pca = (self.pc is None)
                if no_pca:
                    raise ValueError
            except ValueError:
                print("PCA hasn't been computed yet, perfom PCA analysis before doing linear regression or set pca = True")
                print("No linear regression computed")
                return
            
            self.regr_real = linear_model.LinearRegression()
            self.regr_real.fit(self.FFT_pc, self.fft_list[0].real)

            print('Real')
            print('----')
            print('Intercept: \n', self.regr_real.intercept_)
            print('Coefficients: \n', self.regr_real.coef_)

            self.regr_imag = linear_model.LinearRegression()
            self.regr_imag.fit(self.FFT_pc, self.fft_lis[0].imag)

            print('\nImaginary')
            print('---------')
            print('Intercept: \n', self.regr_imag.intercept_)
            print('Coefficients: \n', self.regr_imag.coef_)
            
            self.FFT_real_ac = self.regr_real.predict(self.FFT_pc)  #Why to the whole fft and not to each real and imaginary
            self.FFT_imag_ac = self.regr_imag.predict(self.FFT_pc)
        
        else:
            self.regr_real = linear_model.LinearRegression()
            self.regr_real.fit(self.scaledFFT, self.fft_list[0].real)

            print('Real')
            print('----')
            print('Intercept: \n', self.regr_real.intercept_)
            print('Coefficients: \n', self.regr_real.coef_)

            self.regr_imag = linear_model.LinearRegression()
            self.regr_imag.fit(self.scaledFFT, self.fft_list[0].imag)

            print('\nImaginary')
            print('---------')
            print('Intercept: \n', self.regr_imag.intercept_)
            print('Coefficients: \n', self.regr_imag.coef_)
            
            self.FFT_real_ac = self.regr_real.predict(self.scaledFFT)  #Why to the whole fft and not to each real and imaginary
            self.FFT_imag_ac = self.regr_imag.predict(self.scaledFFT)
            
        
    def fftac_plot(self):
        
        '''
        Displays the Frequency Domain activity model along the original NFFT of the main variable
        '''
        try:
            no_linear = (self.FFT_real_ac is None) and (self.FFT_imag_ac is None)
            if no_linear:
                raise ValueError
        except ValueError:
            print("Linear regression hasn't been computed, run linear_reg()")
            print("No plot displayed")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7,8))
        ax1.plot(self.fgrid, self.fft_list[0].real, label="Total " + self.names[0], color='mediumblue', alpha = 0.8)
        ax1.plot(self.fgrid, self.FFT_real_ac, '--', label="Activity model", color='crimson')
        ax1.legend(loc="best")
        ax1.set_ylabel("Real")
        ax1.set_title("Activity RV from Linear Regression")
        ax2.plot(self.fgrid, self.fft_list[0].imag, color='mediumblue', alpha = 0.8)
        ax2.plot(self.fgrid, self.FFT_imag_ac, '--', color='crimson')
        ax2.set_ylabel("Imaginary")
        ax2.set_xlabel("Frequency")
        plt.tight_layout()
    
    def inverfft(self):
        
        '''
        Transforms the activity model from the frequency domain back to the time domain by applying an inverse NFFT by using Parseval's Theorem
        '''
        
        try:
            no_linear = (self.FFT_real_ac is None) and (self.FFT_imag_ac is None)
            if no_linear:
                raise ValueError
        except ValueError:
            print("Linear regression hasn't been computed, run linear_reg()")
            print("No inversion from frequecy domain calculated")
            return
            
        self.FFT_ac = np.zeros_like(self.fft_list[0])
        self.FFT_ac.real = self.FFT_real_ac
        self.FFT_ac.imag = self.FFT_imag_ac

        # FINUFFT amplitude scaling is a bit ratty, hence "raw"
        self.activity_raw = nufft1d3(2*np.pi*self.fgrid, self.FFT_ac, self.t, isign=1, nthreads=1) / self.N
        #print(np.max(self.activity_raw.real), np.max(self.activity_raw.imag)) # check that the imaginary parts are negligible
        
        # Rescale variance using Parseval's theorem
        variance_ratio = np.sum(np.abs(self.FFT_ac)**2) / np.sum(np.abs(self.fft_list[0])**2)

        print("Raw standard deviation", f"{self.obs[0].std():.4f}")
        print("Variance ratio:", f"{variance_ratio:.4f}")

        # Make activity RVs now have unit variance
        self.activity_scaled = (self.activity_raw.real - np.mean(self.activity_raw.real)) / np.std(self.activity_raw.real)
        #print("Should be 1:", np.var(self.activity_scaled))

        # Calculate the standard deviation of the activity signal
        new_stdev = np.sqrt(variance_ratio * self.obs[0].var())
        self.activity = self.activity_scaled * new_stdev + self.obs[0].mean()
        #print("Should be equal:", f"{np.std(self.activity):.4f}", f"{new_stdev:.4f}")
        print("Inversion fully computed")
        
    def activityplot(self, err = None, clean = False):
        
        '''
        Displays the plot of the activity correction and its residuals
        :param err: Error estimates for the first component to be shown in the plot (must be same shape as :param:`first_com`, default value is None 
        :type err: list, optional
        :param clean: Option to display the residual time series after substracting the acitvity model from the original observation, defualt is False
        :type clean: bool, optional
        '''
        try:
            no_inver = (self.activity is None)
            if no_inver:
                raise ValueError
        except ValueError:
            print("The inversion of the activity model hasn't been computed, run inverfft()")
            print("No plot displayed")
            return
        plt.figure(figsize=(8,6))
        plt.errorbar(self.t, self.first_com, yerr= err, marker='s', ls='none', color='mediumblue', label='Total')
        plt.scatter(self.t, self.activity, color='crimson', marker='o', label='Activity')
        plt.xlabel('Time (days)')
        plt.ylabel(self.names[0])
        plt.title('Activity correction')
        plt.legend(loc='lower left')
        
        if clean == True:
            residuals = self.obs[0] - self.activity
            print("Std dev of clean RV:", np.std(residuals))
            plt.figure(figsize=(8,6))
            #plt.errorbar(self.t, self.obs[0], yerr= err, marker='s', ls='none', color='mediumblue', label='Total')
            plt.scatter(self.t, residuals, color='crimson', marker='o', label='Residuals')
            plt.xlabel('Time (days)')
            plt.ylabel(self.names[0])
            plt.title('Residuals')
            plt.legend(loc='lower left')
        
    def residual_powplot(self):
        
        '''
        Computes the power spectra of the residuals and displays its result
        '''

        try:
            no_inver = (self.activity is None)
            if no_inver:
                raise ValueError
        except ValueError:
            print("The inversion of the activity model hasn't been computed, run inverfft()")
            print("No plot displayed")
            return
        
        self.residuals = self.obs[0] - self.activity
        self.residuals_FFT = nufft1d3(2*np.pi*self.t, self.residuals - np.mean(self.residuals), self.fgrid, \
                                      isign=-1, nthreads=1)
        
        plt.figure(figsize=(8,6))
        plt.plot(self.fgrid, np.abs(self.fft_list[0])**2, label='Total', color='mediumblue', alpha = 0.5)
        plt.plot(self.fgrid, np.abs(self.residuals_FFT)**2,'--', label='Cleaned', color='r')
        plt.xlim(0,np.max(self.fgrid))
        #plt.yscale('log')
        # plt.ylim([10,100000])
        plt.xlabel('Frequency (cycles / day)', fontsize='large')
        plt.ylabel(self.names[0] + ' power spectrum', fontsize='large')
        plt.title('Corrected power Spectra', fontsize =16)
        plt.legend(loc='best')
        plt.tight_layout()