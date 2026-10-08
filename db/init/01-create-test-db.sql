-- Runs once, when the database volume is first created.
-- A separate database for integration tests, so tests never touch app data.
CREATE DATABASE studio_test;
