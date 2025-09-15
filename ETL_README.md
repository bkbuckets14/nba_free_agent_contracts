The ETL step of this problem involves performing some web scraping to get player data, stats data, and free agent contract data and then adding this data to a SQLite database.

The user has two options for doing so:

1) Run the etl.ipynb Jupyter Notebook.
   Running all the cells in this notebook will perform all the necessary steps required for ETL.

2) Running the following commands from the command line:

   rm data.db
   sqlite3 data.db < data.sql
   python3 bk_etl.py

*Note: Running the main.ipynb file will take care of ETL along with the other steps for the full completion of this problem.

Data URLs:
Player Data: https://www.basketball-reference.com/teams/BOS/2021.html *Note: Taking data from 'Roster' Table. One webpage/table for each team and year.
Stats Data: https://www.basketball-reference.com/leagues/NBA_2021_per_game.html *Note: Taking data from 'Player Per Game' Table. One webpage/table for each year.
Free Agent Contract Data: https://www.spotrac.com/nba/free-agents/2021/ *Note: Taking data from main table. One webpage/table for each year.