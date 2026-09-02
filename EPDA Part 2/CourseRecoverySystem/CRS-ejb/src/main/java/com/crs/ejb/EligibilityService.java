package com.crs.ejb;

import com.crs.entity.Student;
import com.crs.entity.StudentCourse;
import com.crs.util.GradeUtil;

import jakarta.ejb.Stateless;
import jakarta.persistence.EntityManager;
import jakarta.persistence.PersistenceContext;
import jakarta.persistence.TypedQuery;
import java.util.ArrayList;
import java.util.List;

/**
 * Handles eligibility checks and enrolment into the next level of study.
 *
 * Progression criteria:
 *   1. CGPA >= 2.0
 *   2. Not more than 3 failed courses
 */
@Stateless
public class EligibilityService {

    private static final double MIN_CGPA        = 2.0;
    private static final int    MAX_FAILED       = 3;

    @PersistenceContext(unitName = "CRSPU")
    private EntityManager em;

    /**
     * Calculates the current CGPA for a student based on their best attempt
     * per course (using the latest attempt's grade for each course).
     */
    public double calculateCGPA(int studentId) {
        List<StudentCourse> enrollments = getLatestAttempts(studentId);
        if (enrollments.isEmpty()) return 0.0;

        double[] gps = new double[enrollments.size()];
        int[]    chs = new int[enrollments.size()];
        for (int i = 0; i < enrollments.size(); i++) {
            StudentCourse sc = enrollments.get(i);
            gps[i] = sc.getGradePoint() != null ? sc.getGradePoint() : 0.0;
            chs[i] = sc.getCourse().getCreditHours();
        }
        return GradeUtil.calculateCGPA(gps, chs);
    }

    /**
     * Returns the number of currently failed courses (last attempt not passed).
     */
    public long countFailedCourses(int studentId) {
        return getLatestAttempts(studentId).stream()
                .filter(sc -> !sc.isPassed())
                .count();
    }

    /**
     * Returns true if the student is eligible to progress to the next level.
     */
    public boolean isEligible(int studentId) {
        return calculateCGPA(studentId) >= MIN_CGPA
                && countFailedCourses(studentId) <= MAX_FAILED;
    }

    /** Returns all active, non-graduated students who are NOT eligible to progress. */
    public List<Student> getIneligibleStudents() {
        List<Student> all = em.createNamedQuery("Student.findAll", Student.class).getResultList();
        List<Student> ineligible = new ArrayList<>();
        for (Student s : all) {
            if ("GRADUATED".equals(s.getStatus())) continue;
            if (!isEligible(s.getStudentId())) ineligible.add(s);
        }
        return ineligible;
    }

    /** Returns all active, non-graduated students who ARE eligible to progress. */
    public List<Student> getEligibleStudents() {
        List<Student> all = em.createNamedQuery("Student.findAll", Student.class).getResultList();
        List<Student> eligible = new ArrayList<>();
        for (Student s : all) {
            if ("GRADUATED".equals(s.getStatus())) continue;
            if (isEligible(s.getStudentId())) eligible.add(s);
        }
        return eligible;
    }

    /**
     * Progresses an eligible student to the next year.
     * If the student is already in Year 4, marks them as GRADUATED instead.
     * Returns true if graduated, false if progressed to next year.
     */
    public boolean progressToNextYear(int studentId) {
        if (!isEligible(studentId)) {
            throw new IllegalStateException("Student " + studentId + " is not eligible to progress.");
        }
        Student s = em.find(Student.class, studentId);
        if (s == null) throw new IllegalArgumentException("Student " + studentId + " not found.");
        if (s.getYearOfStudy() >= 4) {
            s.setStatus("GRADUATED");
        } else {
            s.setYearOfStudy(s.getYearOfStudy() + 1);
        }
        em.merge(s);
        return "GRADUATED".equals(s.getStatus());
    }

    // ---- Private helpers ----

    /**
     * Returns the most recent StudentCourse record per course for a student.
     * Uses a JPQL query that groups by course and picks max attempt.
     */
    private List<StudentCourse> getLatestAttempts(int studentId) {
        String jpql =
            "SELECT sc FROM StudentCourse sc " +
            "WHERE sc.student.studentId = :studentId " +
            "  AND sc.attemptNumber = (" +
            "      SELECT MAX(sc2.attemptNumber) FROM StudentCourse sc2 " +
            "      WHERE sc2.student.studentId = :studentId " +
            "        AND sc2.course.courseId = sc.course.courseId" +
            "  )";
        TypedQuery<StudentCourse> q = em.createQuery(jpql, StudentCourse.class);
        q.setParameter("studentId", studentId);
        return q.getResultList();
    }
}
