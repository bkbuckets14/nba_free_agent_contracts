#This program will scrape player data, stats data, and free agent contract data from internet and add it to sqlite database.
#Before running this program, be sure to run the following two commands on a command line:
#    rm data.db
#    sqlite3 data.db < data.sql

#Data URLS
#Player Data: https://www.basketball-reference.com/teams/BOS/2021.html *Note: Taking data from 'Roster' Table. One webpage/table for each team and year.
#Stats Data: https://www.basketball-reference.com/leagues/NBA_2021_per_game.html *Note: Taking data from 'Player Per Game' Table. One webpage/table for each year.
#Free Agent Contract Data: https://www.spotrac.com/nba/free-agents/2021/ *Note: Taking data from main table. One webpage/table for each year.

import requests
from bs4 import BeautifulSoup
import sqlite3

from name_errors import name_errors_dict

YEARS = ['2019', '2020', '2021']

TEAMS = ['BOS', 'BRK', 'NYK', 'PHI', 'TOR',
         'CLE', 'IND', 'CHI', 'MIL', 'DET',
         'MIA', 'CHO', 'ORL', 'ATL', 'WAS',
         'NOP', 'DAL', 'SAS', 'MEM', 'HOU',
         'DEN', 'MIN', 'UTA', 'OKC', 'POR',
         'LAL', 'LAC', 'GSW', 'SAC', 'PHO']

def get_player_data(years, teams):
    #this function will access the player information from each team in teams for each year in years
    
    player_id_dict = {}
    id_count = 1
    data = []

    for year in years:
        for team in teams:
            #this section gets the statistics table from Basketball-Reference.com
            web = requests.get('https://www.basketball-reference.com/teams/'+team+'/'+year+'.html').text
            soup = BeautifulSoup(web, 'lxml')
            table = soup.find('tbody').find_all('tr')
            
            #each row corresponds to one player's information
            for row in table:
                temp = row.find_all('td')
                temp = [x.text for x in temp]
                #if a player is already in player_id_dict (previous season or other team), we don't add them again
                if temp[0] in player_id_dict:
                    continue
                player_id_dict[temp[0]] = id_count #generate id for each player
                temp[2] = temp[2].split('-')
                if temp[6] == 'R': #change rookie to 0 years in league
                    temp[6] = '0'
                if temp[7] == '': #handles players who didn't go to college
                    temp[7] = 'None'
                data.append((id_count, temp[0], temp[1], (int(temp[2][0])*12 + int(temp[2][1])), int(temp[3]),
                             temp[4], temp[5].upper(), (int(year) - int(temp[6])), temp[7]))
                id_count+=1
                    
    return player_id_dict, data


def get_stats_data(years, player_id_dict):
    #this function will access the individual stats for the entire NBA for each year in years
    
    data = [['empty', 'empty', 'empty']]

    for year in years:
        #this section gets the statistics table from Basketball-Reference.com
        web = requests.get('https://www.basketball-reference.com/leagues/NBA_'+year+'_per_game.html').text
        soup = BeautifulSoup(web, 'lxml')
        table = soup.find('tbody').find_all('tr')
        
        #each row corresponds to one player's stats
        for row in table:
            temp = row.find_all('td')
            temp = [x.text for x in temp]
            
            #this if block ignores rows not associated with a player and handles the situation where a player played
            #for more than one team in a season.
            if temp == [] or temp[0]==data[-1][2]: 
                continue
            temp.pop(1) #get rid of the position
            for i in [1, 3, 4]: #set these statistics to integers
                temp[i] = int(temp[i])
            for i in range(5, len(temp)): #set these statistics to floats
                try:
                    temp[i] = float(temp[i])
                except:
                    temp[i] = 0
                #try except block handles case where player has undefined percentage stats, sets to 0
            
            
            #gets player_id from player_id_dict, except block handles case where players name is different among pages
            try:
                p_id = player_id_dict[temp[0]]
            except KeyError:
                if temp[0] in name_errors_dict:
                    p_id = player_id_dict[name_errors_dict[temp[0]]]
                if ' '.join(temp[0].split(' ')[:2]) in player_id_dict:
                    p_id = player_id_dict[' '.join(temp[0].split(' ')[:2])]
                else:
                    continue
                
            temp.insert(0, int(year))
            temp.insert(0, p_id)
            
            
            data.append(tuple(temp))
    
    data.pop(0)
    
    return data


def get_free_agent_data(years, player_id_dict):
    #this function will access the contract information for all the free agents for each year in years
    data = []
    
    for year in years:
        #this section gets the statistics table from Spotrac.com
        web = requests.get('https://www.spotrac.com/nba/free-agents/'+year+'/').text
        soup = BeautifulSoup(web, 'lxml')
        table = soup.find('tbody').find_all('tr')
        
        #each row corresponds to one contract
        for row in table:
            temp = row.find_all('td')
            temp = [x.text.strip() for x in temp]
            if temp[7]=='0-': #disregard free agents who didn't sign contracts
                continue
            temp.pop(1) #get rid of the position
            for i in range(len(temp)): #remove extraneous characters
                for char in ['$', '>', ',']:
                    temp[i] = temp[i].replace(char,'')
            temp[1] = float(temp[1])
            for i in range(5,8):
                temp[i] = int(temp[i])
            
            #gets player_id from player_id_dict, except block handles case where players name is different among pages
            try:
                p_id = player_id_dict[temp[0]]
            except KeyError:
                if temp[0] in name_errors_dict:
                    p_id = player_id_dict[name_errors_dict[temp[0]]]
                if ' '.join(temp[0].split(' ')[:2]) in player_id_dict:
                    p_id = player_id_dict[' '.join(temp[0].split(' ')[:2])]
                else:
                    continue
                
            temp.insert(0, int(year))
            temp.insert(0, p_id)

            if temp[5]==temp[6]:
                change_team = 0
            else:
                change_team = 1
            temp.insert(7, change_team) 
            
            data.append(tuple(temp))
    
    return data


player_id_dict, player_data = get_player_data(YEARS, TEAMS)

stats_data = get_stats_data(YEARS, player_id_dict)

free_agents_data = get_free_agent_data(YEARS, player_id_dict)

#connecting to the database
con = sqlite3.connect('data.db')

#populate players table
con.executemany('INSERT INTO players (id, name, position, height, weight, birthday, country, rookie_year, college) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', player_data);
con.commit()

#populate stats table
con.executemany('INSERT INTO stats (id, year, name, age, team, games, games_started, minutes, fg, fga, fg_per, three_fg, three_fga, three_fg_per, two_fg, two_fga, two_fg_per, efg_per, ft, fta, ft_per, orb, drb, trb, ast, stl, blk, tov, pfl, pts) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', stats_data);
con.commit()

#populate contracts table
con.executemany('INSERT INTO contracts (id, year, name, age, type, old_team, new_team, chg_team, length, total_dollars, avg_dollars) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', free_agents_data);
con.commit()

#closing the connection to the database
con.close()
