package com.crs.ejb;

import com.crs.entity.AssessmentComponent;
import com.crs.entity.Course;
import com.crs.entity.StudentCourse;
import com.crs.util.GradeUtil;

import jakarta.ejb.Stateless;
import jakarta.persistence.EntityManager;
import jakarta.persistence.PersistenceContext;
import jakarta.persistence.TypedQuery;
import java.time.LocalDate;
import java.util.List;

@Stateless
public class CourseService {

    @PersistenceContext(unitName = "CRSPU")
    private EntityManager em;

    // ---- Courses ----

    public void createCourse(Course course) {
        em.persist(course);
    }

    public Course findById(int courseId) {
        return em.find(Course.class, courseId);
    }

    public List<Course> findAll() {
        return em.createNamedQuery("Course.findAll", Course.class).getResultList();
    }

    public void updateCourse(Course course) {
        em.merge(course);
    }

    public void deactivateCourse(int courseId) {
        Course c = findById(courseId);
        if (c != null) { c.setActive(false); em.merge(c); }
    }

    // ---- Assessment Components ----

    public void addComponent(AssessmentComponent component) {
        em.persist(component);
    }

    public List<AssessmentComponent> getComponents(int courseId) {
        TypedQuery<AssessmentComponent> q =
                em.createNamedQuery("AssessmentComponent.findByCourse", AssessmentComponent.class);
        q.setParameter("courseId", courseId);
        return q.getResultList();
    }

    public void removeComponent(int componentId) {
        AssessmentComponent c = em.find(AssessmentComponent.class, componentId);
        if (c != null) em.remove(c);
    }

    // ---- Enrollments ----

    /**
     * Enrols a student in a course for a given semester/year at a specific attempt number.
     * Enforces the 3-attempt policy: throws if attemptNumber > 3.
     */
    public StudentCourse enrol(int studentId, int courseId, int semester,
                               int yearOfStudy, int attemptNumber) {
        if (attemptNumber > 3) {
            throw new IllegalArgumentException("Maximum 3 attempts per course exceeded.");
        }

        StudentCourse sc = new StudentCourse();
        sc.setStudent(em.getReference(com.crs.entity.Student.class, studentId));
        sc.setCourse(em.getReference(Course.class, courseId));
        sc.setSemester(semester);
        sc.setYearOfStudy(yearOfStudy);
        sc.setAttemptNumber(attemptNumber);
        sc.setEnrollmentDate(LocalDate.now());
        em.persist(sc);
        return sc;
    }

    /**
     * Records a final grade for an enrollment and marks it passed/failed.
     */
    public void recordGrade(int enrollmentId, String grade) {
        StudentCourse sc = em.find(StudentCourse.class, enrollmentId);
        if (sc == null) throw new IllegalArgumentException("Enrollment not found: " + enrollmentId);
        sc.setGrade(grade);
        sc.setGradePoint(GradeUtil.toGradePoint(grade));
        sc.setPassed(GradeUtil.isPassed(grade));
        em.merge(sc);
    }
}
