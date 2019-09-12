CREATE TABLE crimes

(
  ID int PRIMARY KEY,
  Case_Number varchar(64),
  Date varchar(64),
  Block varchar(64),
  IUCR varchar(64),
  Primary_Type varchar(64),
  Description varchar(64),
  Location_Description varchar(64),
  Arrest boolean,
  Domestic boolean ,
  Beat smallint,
  District smallint,
  Ward smallint,
  Community_Area smallint,
  FBI_Code varchar(64),
  X_Coordinate float,
  Y_Coordinate float,
  Year int,
  Updated_On varchar(64),
  Latitude float,
  Longitude float,
  Location varchar(64)

);
