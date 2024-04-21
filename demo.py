"""
demo.py
Spring 23 PJW

Demonstrate estimation and Monte Carlo analysis for fuel use by
natural gas power plants.

Note: this is set up as a single script so that it can be linked
into the quick reference file easily. Under normal circumstances
it would be 4 to 6 shorter scripts.
"""

import pandas as pd
import statsmodels.api as sm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob

plt.rcParams['figure.dpi'] = 300

#
#  Data files from EIA's Monthly Energy Review (MER)
#
#    MER_T07_03B.csv   # fuel use, electric power
#    MER_T09_09.csv    # fuel cost, electric power
#    MER_TA4.csv       # heat content, gas
#    MER_TA5.csv       # heat content, coal
#    MER_T11_06.csv    # CO2 emissions
#

mer_files = glob.glob("eia/MER_T*.csv")

#
#  Additional EIA data with macro variables
#

macro_file = 'eia/Table_C1.xlsx'

#
#  Values appearing in EIA files that should be treated as missing
#

eia_miss = ['Not Available',
            'Not Meaningful',
            'Withheld',
            'Not Applicable']

#%% set up read_mer_file
#
#  Function for reading an MER file, which are in long format
#

def read_mer_file(fname,missing):

    #  Read the CSV file

    raw = pd.read_csv(
        fname,
        dtype={'YYYYMM':str},
        na_values=missing,
        )

    #  Collect information about each series

    info = raw[['MSN','Description','Unit']].drop_duplicates()
    info.set_index('MSN',inplace=True)

    #  Find annual data, which has 13 for the month

    is_ann = raw['YYYYMM'].str.endswith('13')

    data = raw[ is_ann ].copy()
    data['Year'] = data['YYYYMM'].str[:4].astype(int)
    data = data[['MSN','Value','Year']]

    print(f'File: {fname}, {len(info)} variables, {len(data)} records')

    return (info,data)

#%% read MER files
#
#  Go through the MER list and stack all the data into dataframes
#  of information and actual data
#

info_list = []
data_list = []

#
#  Read the files and add the dataframes to the lists
#

for fname in mer_files:

    #  Read the file

    (cur_info, cur_data) = read_mer_file(fname, eia_miss)

    #  Add to the lists

    info_list.append(cur_info)
    data_list.append(cur_data)

#
#  Now do the actual concatenation
#

info = pd.concat(info_list)
data = pd.concat(data_list)

#
#  Say a little about what we found
#

print( '\nSample info:' )
print( info.head(10) )

print( '\nSample data:' )
print( data.head(10) )

#%% variable info
#
#  Now select the data we'll actually need on coal and gas use.
#  Start by making a list of key variables with friendlier names.
#

map_names = {
    'CLL1PUS':'coal_kton',
    'NGL1PUS':'gas_bcf',
    'CLERDUS':'coal_price_mmbtu',
    'NGERDUS':'gas_price_mmbtu',
    'NGMPKUS':'gas_btu_cf',
    'CLPRKUS':'coal_mmbtu_ton',
    'CLEIEUS':'coal_co2_mmt',
    'NNEIEUS':'gas_co2_mmt',
     }

#
#  Print information about what we're keeping for confirmation
#

keepers = map_names.keys()

for old in keepers:
    what = info.loc[old]['Description']
    unit = info.loc[old]['Unit']
    new = map_names[old]
    print(f'\n{old} -> {new}\n   {what}\n   {unit}')

#%% filter and pivot
#
#  Filter the data down to only the rows we want.
#

select = data[ data['MSN'].isin(keepers) ]

#
#  Now unstack it for convenience. Use .pivot() since the information
#  to use for unstacking is in a column rather than the index.
#

uns = select.pivot( index='Year', columns='MSN', values='Value' )

#
#  Rename the columns
#

uns.rename(columns=map_names, inplace=True)

#
#  Drop years with missing data: before 1972 and after latest year
#

has_na = uns.isna().any(axis='columns')

print( '\nDropping years with missing data:')
print( uns[has_na].index )

uns = uns[ has_na == False ]

#%% emissions coefficients
#
#  Calculate fuel use in million BTU to match price variables. Scale
#  coal up to tons and gas up to cubic feet, then multiply by heat
#  content. Use intermediate variables along the way to spell out the
#  calculation.
#

coal_tons    = uns['coal_kton' ]*1e3  # kilotons to tons
gas_cf       = uns['gas_bcf'   ]*1e9  # billion cf to cf
gas_mmbtu_cf = uns['gas_btu_cf']/1e6  # btu per cf to mmbtu per cf

uns['coal_mmbtu'] = coal_tons * uns['coal_mmbtu_ton']
uns['gas_mmbtu' ] = gas_cf    * gas_mmbtu_cf

#
#  Calculate tons of CO2 emitted for a million BTU of fuel
#

coal_co2_mt = uns['coal_co2_mmt']*1e6  # million tons to tons
gas_co2_mt  = uns['gas_co2_mmt' ]*1e6  # million tons to tons

uns['coal_mt_mmbtu'] = coal_co2_mt/uns['coal_mmbtu']
uns['gas_mt_mmbtu' ] = gas_co2_mt /uns['gas_mmbtu' ]

#
#  Print as kg/mmbtu for checking
#

print( '\nMean emissions coefficients, kg/mmbtu:\n' )
print( 1e3*uns[['coal_mt_mmbtu','gas_mt_mmbtu']].mean() )

#%% read macro data
#
#  Now read the macro data.
#

g_raw = pd.read_excel(
    macro_file,
    sheet_name='Annual Data',
    skiprows=lambda x: x in range(0,10) or x==11,
    na_values=eia_miss,
    index_col='Year')

#
#  Extract the variables we'll need. For reference, prices are in
#  2012 dollars.
#

pop  = g_raw['Total Resident Population, United States']
gdp  = g_raw['U.S. Gross Domestic Product, Real']
defl = g_raw['U.S. Gross Domestic Product Implicit Price Deflator']

#%% streamline
#
#  Now build a streamlined dataframe for estimation. Keep only the
#  variables we'll need and shorten names a bit.
#

res = pd.DataFrame()

#  Measure quantities in mmbtu

res['q_coal'] = uns['coal_mmbtu']
res['q_gas' ] = uns['gas_mmbtu']

#  Measure prices in real dollars per mmbtu

res['p_coal'] = uns['coal_price_mmbtu']/defl
res['p_gas' ] = uns['gas_price_mmbtu' ]/defl

#  For convenience later, keep CO2 emissions coefficients

res['co2_coal'] = uns['coal_mt_mmbtu']
res['co2_gas' ] = uns['gas_mt_mmbtu' ]

#  Keep macro variables

res['pop'] = pop
res['gdp'] = gdp

#  Write it all out for reference, or for using with Stata

res.to_csv('fuels.csv')

#%% set up estimation
#
#  Now estimate the gas demand equation.
#
#  First, compute natural logs of several variables.
#

for v in ['q_gas','p_gas','p_coal','gdp']:
    res[f'ln_{v}'] = res[v].apply(np.log)

#
#  For clarity, set up the dependent and independent variables. Will
#  regress log of gas Q on logs of gas and coal prices and log of GDP
#

dep_var  = 'ln_q_gas'
ind_vars = ['ln_p_gas','ln_p_coal','ln_gdp']

#
#  Set things up for the statsmodels API
#

Y = res[dep_var]

X = res[ind_vars]
X = sm.add_constant(X)

model = sm.OLS(Y,X)

#%% estimate
#
#  Now do the actual estimation and print a summary
#

results = model.fit()

print( results.summary() )

#%% get results
#
#  Extract and print key elements of the results
#

est = results.params
cov = results.cov_params()

print( '\nExtracted parameter estimates:\n' )
print( est )

print( '\nExtracted parameter covariance matrix:\n' )
print( cov )

std_errs = np.diag(cov)**0.5
std_err_series = pd.Series( data=std_errs, index=cov.index )

print( '\nSquare root of covariance diagonal:\n' )
print( std_err_series )

#%% draw sample
#
#  Now set up the Monte Carlo analysis.
#
#  Use the estimation results to generate a large number of
#  parameter vectors that are consistent with the joint distribution
#  of the parameters.
#

#
#  Set the initial seed for replicability.
#

np.random.seed(0)

#
#  Draw the parmeters and make them into a dataframe. The order of
#  the columns in the draw match the order of the parameter estimates,
#  so use the parameter names for the column names.
#

draws_raw = np.random.multivariate_normal(est,cov,size=10000)

draws = pd.DataFrame( columns=est.index, data=draws_raw )

#
#  Show some information about the draws
#

print( '\nSample draws:\n')
print( draws.head() )

compare = pd.DataFrame(
    index =['estimates','draw mean'],
    data = [est, draws.mean()])

print( '\nComparing with estimates:\n')
print( compare )

#%% plot parameters
#
#  Plot the distribution of two key parameters
#

fig,ax = plt.subplots()

fig.suptitle('Distribution of Parameters\n75% and 90% ellipsoids shown in red')

#
#  First, draw a two-dimensional histogram
#

sns.histplot(
    data=draws,
    x='ln_p_gas',
    y='ln_p_coal',
    ax=ax)

#
#  Now overlay three confidence ellipsoids for 75% and 90%
#

sns.kdeplot(
    data=draws,
    x='ln_p_gas',
    y='ln_p_coal',
    levels=[0.10, 0.25],
    color='red',
    ax=ax)

ax.set_ylabel('Coefficient on log of coal price')
ax.set_xlabel('Coefficient on log of gas price')

fig.tight_layout()
fig.savefig('dist_par.png')

#%% set up analysis
#
#  Now set up the policy analysis.
#
#  Start by setting up a function that will calculate the Q of gas
#  for a given set of independent variables at each of the
#  parameter vectors in the set of draws.
#

def predict_q_gas(indep_vars, par_draws):

    #  Calculate individual beta*X terms
    #     b0*x0, b1*x1, ...

    beta_X = par_draws*indep_vars

    #  Sum them up to get the predicted log of Q of gas
    #     y = b0*x0 + b1*x1 ...

    ln_q_gas = beta_X.sum(axis='columns')

    #  Convert from log to Q

    result = ln_q_gas.apply(np.exp)

    #  Return the result

    return result

#%% set up cases
#
#  Case 1: BAU variables
#

means = res.mean()

ind_vars_1 = means[ind_vars]
ind_vars_1['const'] = 1

#  For convenience, reconstruct prices

p1_gas  = np.exp( ind_vars_1['ln_p_gas' ] )
p1_coal = np.exp( ind_vars_1['ln_p_coal'] )

#
#  Case 2: variables under the policy shock
#

tax = 50

tax_gas  = tax*means['co2_gas' ]
tax_coal = tax*means['co2_coal']

#  Treat the fuels as perfectly elastic and calculate new buyer prices

p2_gas  = p1_gas  + tax_gas
p2_coal = p1_coal + tax_coal

#  Counterfactual: replace historical data with values adjusted by tax

ind_vars_2 = ind_vars_1.copy()
ind_vars_2['ln_p_gas' ] = np.log(p2_gas)
ind_vars_2['ln_p_coal'] = np.log(p2_coal)

#%% describe cases
#
#  Describe the cases briefly
#

pct_gas  = 100*(p2_gas /p1_gas -1)
pct_coal = 100*(p2_coal/p1_coal-1)

print( '\nCO2 tax per mt:',tax)
print( '\nImpact on gas price (base plus tax): ' )
print( f'   {p1_gas:.2f} + {tax_gas:.2f} = {p2_gas:.2f}, {pct_gas:.1f}%')
print( '\nImpact on coal price (base plus tax):' )
print( f'   {p1_coal:.2f} + {tax_coal:.2f} = {p2_coal:.2f}, {pct_coal:.1f}%')

compare = pd.DataFrame(
    index=['bau','tax'],
    data=[ind_vars_1,ind_vars_2])

print( '\nComparing independent variables:\n' )
print( compare )

#%% do the calculations
#
#  Now calculate both cases: bau and the policy analysis
#

case1 = predict_q_gas(ind_vars_1, draws)
case2 = predict_q_gas(ind_vars_2, draws)

#
#  Impact on gas use? Rises despite the tax.
#

pct = 100*(case2/case1 - 1)

pct_stats = pct.describe(percentiles=[0.05,0.95])
print( '\nPercent changes in Qg:\n')
print( pct_stats )

print( '\nConfidence Interval for Change in Q gas, %')
print( pct_stats[['5%','95%']].round(0) )

#
#  Revenue raised via the tax on gas in billions
#

rev = tax_gas*case2/1e9

rev_stats = rev.describe(percentiles=[0.05,0.95])
print( '\nRevenue in billions:\n')
print( rev_stats )

print( '\nConfidence Interval for Revenue, B$')
print( rev_stats[['5%','95%']].round(0) )

#
#  Now plot the revenue
#

fig,ax = plt.subplots(layout='constrained')

fig.suptitle('Distribution of Revenue\n90% confidence interval shown in red')

sns.histplot(rev,stat='percent',ax=ax)

ax.set_xlabel( 'Billions of dollars' )
ax.set_xlim(left=0)

ax.axvline(rev_stats['5%'],c='r')
ax.axvline(rev_stats['mean'],c='orange')
ax.axvline(rev_stats['50%'],c='cyan')
ax.axvline(rev_stats['95%'],c='r')

fig.supxlabel('Note: cyan=median, orange=mean',fontsize='medium')

fig.savefig('dist_rev.png')

#%% compare impacts
#
#  Finally, compare impacts from changes in gas and coal prices on
#  gas consumption. Case 3 captures only gas price effects and
#  Case 4 captures only coal price effects.
#

#
#  Function to compute a counterfactual using BAU plus values of setvar
#

def get_case_q_pct(setvar):

    ind_vars = ind_vars_1.copy()
    ind_vars[setvar] = ind_vars_2[setvar]

    res   = predict_q_gas(ind_vars, draws)
    pct   = 100*(res/case1 - 1)

    return pct

#
#  Build a dataframe of results
#

allcases = pd.DataFrame()
allcases['2, tax'] = pct
allcases['3, p gas only'] = get_case_q_pct('ln_p_gas')
allcases['4, p coal only'] = get_case_q_pct('ln_p_coal')

#%%
#
#  Report some summary statistics
#

stats = allcases.describe(percentiles=[0.05,0.95])

print( '\nPercentage changes in Qg:\n')
print(stats)

#%%
#
#  Plot the distributions
#

stack = allcases.stack().reset_index(1)
stack = stack.rename(columns={'level_1':'Case',0:'pct'})

(m2,m3,m4) = stats.loc['50%'].round(0).astype(int)

g = sns.displot(data=stack,x='pct',hue='Case',kind='hist')
g.set_axis_labels('Percent Change in Q of gas','Count')
g.fig.suptitle('Decomposing the Effects of Gas and Coal Prices')
g.fig.supxlabel(f'Medians: 2={m2}%, 3={m3}%, 4={m4}%',fontsize='medium')
g.fig.tight_layout()
g.fig.savefig('decomp.png')
