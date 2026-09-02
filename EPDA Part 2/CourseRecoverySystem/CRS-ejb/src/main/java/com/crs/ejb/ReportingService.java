package com.crs.ejb;

import com.crs.entity.Student;
import com.crs.entity.StudentCourse;
import com.crs.util.GradeUtil;

import jakarta.ejb.Stateless;
import jakarta.persistence.EntityManager;
import jakarta.persistence.PersistenceContext;
import jakarta.persistence.TypedQuery;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Generates academic performance reports by semester and year.
 */
@Stateless
public class ReportingService {

    @PersistenceContext(unitName = "CRSPU")
    private EntityManager em;

    /**
     * Returns all enrollment records for a student in a given semester/year,
     * grouped as a ready-to-display report.
     */
    public SemesterReport generateSemesterReport(int studentId, int semester, int yearOfStudy) {
        Student student = em.find(Student.class, studentId);
        if (student == null) throw new IllegalArgumentException("Student not found.");

        String jpql =
            "SELECT sc FROM StudentCourse sc " +
            "WHERE sc.student.studentId = :studentId " +
            "  AND sc.semester = :semester " +
            "  AND sc.yearOfStudy = :yearOfStudy " +
            "ORDER BY sc.course.courseCode";

        TypedQuery<StudentCourse> q = em.createQuery(jpql, StudentCourse.class);
        q.setParameter("studentId", studentId);
        q.setParameter("semester", semester);
        q.setParameter("yearOfStudy", yearOfStudy);
        List<StudentCourse> rows = q.getResultList();

        double[] gps = rows.stream().mapToDouble(sc -> sc.getGradePoint() != null ? sc.getGradePoint() : 0.0).toArray();
        int[]    chs = rows.stream().mapToInt(sc -> sc.getCourse().getCreditHours()).toArray();
        double   cgpa = GradeUtil.calculateCGPA(gps, chs);

        return new SemesterReport(student, semester, yearOfStudy, rows, cgpa);
    }

    /**
     * Returns semester reports for all semesters a student has attended,
     * keyed by "Year X – Semester Y".
     */
    public Map<String, SemesterReport> generateFullReport(int studentId) {
        String jpql =
            "SELECT DISTINCT sc.yearOfStudy, sc.semester FROM StudentCourse sc " +
            "WHERE sc.student.studentId = :studentId " +
            "ORDER BY sc.yearOfStudy, sc.semester";

        List<Object[]> periods = em.createQuery(jpql, Object[].class)
                                   .setParameter("studentId", studentId)
                                   .getResultList();

        Map<String, SemesterReport> report = new LinkedHashMap<>();
        for (Object[] row : periods) {
            int year = (int) row[0];
            int sem  = (int) row[1];
            String key = "Year " + year + " – Semester " + sem;
            report.put(key, generateSemesterReport(studentId, sem, year));
        }
        return report;
    }

    // ---- Inner DTO ----

    public static class SemesterReport {
        private final Student             student;
        private final int                 semester;
        private final int                 yearOfStudy;
        private final List<StudentCourse> enrollments;
        private final double              cgpa;

        public SemesterReport(Student student, int semester, int yearOfStudy,
                              List<StudentCourse> enrollments, double cgpa) {
            this.student     = student;
            this.semester    = semester;
            this.yearOfStudy = yearOfStudy;
            this.enrollments = enrollments;
            this.cgpa        = cgpa;
        }

        public Student             getStudent()     { return student; }
        public int                 getSemester()    { return semester; }
        public int                 getYearOfStudy() { return yearOfStudy; }
        public List<StudentCourse> getEnrollments() { return enrollments; }
        public double              getCgpa()        { return cgpa; }
    }
}
