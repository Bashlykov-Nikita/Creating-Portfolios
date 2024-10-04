# Creating-Portfolio
### Description:
This tool lets you build different kinds of investment portfolios based on past company performance data. You can create portfolios like:

 - Maximum Sharpe Ratio - a portfolio that aims for the highest possible returns for a given level of risk.
 - Global Minimum Variance - a portfolio with the smallest possible fluctuations in value.
 - Equally Weighted - a portfolio where each stock has the same percentage.
 - Cap Weighted - a portfolio where each stock's weight is based on its market value.
 - Equal Risk Contribution - a portfolio where each stock contributes equally to the overall portfolio risk.

In this project, historical data from companies included in well-known indices is used as an example. More details on how this data was obtained can be found [here](https://github.com/Bashlykov-Nikita/Companies-Returns).

### Features:
 1) Generates .csv files containing weights of companies in various portfolios.
 2) Calculates covariance matrices and expected returns using different methods:

    For covariance:  
    - Sample: Uses only past data (100% based on what happened).
    - [Constant Correlation](https://www.jstor.org/stable/2328653): A simpler method that assumes all stocks move together equally (100% based on assumptions).
    - [Shrinkage](http://www.ledoit.net/honey.pdf): A mix of the two above, you can choose how much to rely on each.

    For expected returns:  
    - Average: Uses the simple average of past returns.
    - Exponentially Weighted Average: Gives more weight to recent returns.
    
3) Generates .csv files with backtesting of all portfolios on historical data using a rolling window.
4) (In the works) [Black-Litterman Model](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1117574)
