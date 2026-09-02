package com.crs.ejb;

import com.crs.entity.Student;
import com.crs.entity.StudentCourse;

import jakarta.ejb.Stateless;
import jakarta.persistence.EntityManager;
import jakarta.persistence.PersistenceContext;
import jakarta.persistence.TypedQuery;
import java.util.List;

@Stateless
public class StudentService {

    @PersistenceContext(unitName = "CRSPU")
    private EntityManager em;

    public void createStudent(Student student) {
        em.persist(student);
    }

    public Student findById(int studentId) {
        return em.find(Student.class, studentId);
    }

    public Student findByStudentNumber(String studentNumber) {
        TypedQuery<Student> q = em.createNamedQuery("Student.findByNumber", Student.class);
        q.setParameter("studentNumber", studentNumber);
        List<Student> r = q.getResultList();
        return r.isEmpty() ? null : r.get(0);
    }

    public List<Student> findAll() {
        return em.createNamedQuery("Student.findAll", Student.class).getResultList();
    }

    public void updateStudent(Student student) {
        em.merge(student);
    }

    public void deactivateStudent(int studentId) {
        Student s = findById(studentId);
        if (s != null) {
            s.setActive(false);
            em.merge(s);
        }
    }

    public StudentCourse findEnrollmentById(int enrollmentId) {
        return em.find(StudentCourse.class, enrollmentId);
    }

    /** All enrollment records for a student (all semesters, all attempts). */
    public List<StudentCourse> getEnrollments(int studentId) {
        TypedQuery<StudentCourse> q = em.createNamedQuery("StudentCourse.findByStudent", StudentCourse.class);
        q.setParameter("studentId", studentId);
        return q.getResultList();
    }

    /** Only the failed (is_passed = false) enrollment records for a student. */
    public List<StudentCourse> getFailedCourses(int studentId) {
        TypedQuery<StudentCourse> q = em.createNamedQuery("StudentCourse.findFailedByStudent", StudentCourse.class);
        q.setParameter("studentId", studentId);
        return q.getResultList();
    }
}
