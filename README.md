# Example: Estimation and Monte Carlo Analysis

## Summary

The **demo.py** file in this repository shows how to run a regression using the `statsmodels` module and then how to carry out Monte Carlo analysis to generate the confidence interval for a key variable in a policy experiment.

The example regression is a very simple model of the demand for natural gas by electric utilities. It gives the log of the natural gas (`Qg`) consumed by power producers as a function of the log of the real price of natural gas (`Pg`), the log of the real price of coal (`Pc`), and the log of real GDP (`GDP`):

        ln(Qg) = b0 + b1*ln(Pg) + b2*ln(Pc) + b3*ln(GDP) + e

The Monte Carlo analysis focuses on calculating the **90% confidence interval** (CI) for the mean **revenue** that would be generated on natural gas by a $50 per ton tax on carbon dioxide applied to both gas and coal. Revenue on gas is given by:

        Rev = T*Cg*Qg

where `T` is the tax in dollars per metric ton of CO2, `Qg` is natural gas consumption in million BTU (mmBTU, see Tip 1), and `Cg` is a coefficient giving tons of CO2 emitted for each mmBTU of natural gas used. Revenue on coal is omitted for simplicity but in a full analysis it would be included as well.

The analysis here determines the CI for revenue accounting for uncertainty in the parameter estimates but holding the residuals, `e`, at 0 (hence it is the CI for mean revenue). Finally, to keep things simple, the supplies of gas and coal are assumed to be perfectly elastic.

## Input Data

All data came from the US Energy Information Administration (EIA). The input files are included in the repository in the `eia` subdirectory.

## Deliverables

**None**. This is an example only and there's **nothing due**.

## Instructions

1. Browse the demo.py to see what techniques it demonstrates.

## Tips

1. In the electric power sector, fossil fuels are usually priced and traded according to their energy content in millions of British Thermal Units (abbreviated mmBTU). This analysis follows that approach: in the model `Qg` is measured in mmBTU and `Pg` and `Pc` are in dollars per mmBTU.
