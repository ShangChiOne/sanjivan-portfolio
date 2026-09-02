USE crs_db;

-- Sync with existing is_active values
UPDATE students SET status = 'INACTIVE' WHERE is_active = 0 AND student_id > 0;
UPDATE students SET status = 'ACTIVE'   WHERE is_active = 1 AND student_id > 0;
