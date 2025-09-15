DROP TABLE IF EXISTS player;
CREATE TABLE players (
    id INTEGER PRIMARY KEY,
    name STRING,
    position STRING,
    height INTEGER,
    weight INTEGER,
    birthday STING,
    country STRING,
    rookie_year INTEGER,
    college STRING
);

DROP TABLE IF EXISTS stats;
CREATE TABLE stats (
    id INTEGER,
    year INTEGER,
    name STRING,
    age INTEGER,
    team STRING,
    games INTEGER,
    games_started INTEGER,
    minutes FLOAT,
    fg FLOAT,
    fga FLOAT,
    fg_per FLOAT,
    three_fg FLOAT,
    three_fga FLOAT,
    three_fg_per FLOAT,
    two_fg FLOAT,
    two_fga FLOAT,
    two_fg_per FLOAT,
    efg_per FLOAT,
    ft FLOAT,
    fta FLOAT,
    ft_per FLOAT,
    orb FLOAT,
    drb FLOAT,
    trb FLOAT,
    ast FLOAT,
    stl FLOAT,
    blk FLOAT,
    tov FLOAT,
    pfl FLOAT,
    pts FLOAT
);

DROP TABLE IF EXISTS contracts;
CREATE TABLE contracts (
    id INTEGER,
    year INTEGER,
    name STRING,
    age FLOAT,
    type STRING,
    old_team STRING,
    new_team STRING,
    chg_team INT,
    length INT,
    total_dollars INT,
    avg_dollars INT
);

