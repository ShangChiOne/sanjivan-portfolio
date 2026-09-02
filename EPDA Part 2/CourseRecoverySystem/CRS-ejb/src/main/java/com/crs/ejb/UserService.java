package com.crs.ejb;

import com.crs.entity.User;
import com.crs.util.PasswordUtil;

import jakarta.ejb.Stateless;
import jakarta.persistence.EntityManager;
import jakarta.persistence.PersistenceContext;
import jakarta.persistence.TypedQuery;
import java.util.List;

@Stateless
public class UserService {

    @PersistenceContext(unitName = "CRSPU")
    private EntityManager em;

    // ---- Authentication ----

    /**
     * Returns the User if credentials are valid, otherwise null.
     */
    public User login(String username, String plainPassword) {
        String hash = PasswordUtil.hash(plainPassword);
        TypedQuery<User> q = em.createNamedQuery("User.authenticate", User.class);
        q.setParameter("username", username);
        q.setParameter("password", hash);
        List<User> result = q.getResultList();
        return result.isEmpty() ? null : result.get(0);
    }

    // ---- CRUD ----

    public void createUser(User user, String plainPassword) {
        user.setPassword(PasswordUtil.hash(plainPassword));
        em.persist(user);
    }

    public User findById(int userId) {
        return em.find(User.class, userId);
    }

    public User findByUsername(String username) {
        TypedQuery<User> q = em.createNamedQuery("User.findByUsername", User.class);
        q.setParameter("username", username);
        List<User> r = q.getResultList();
        return r.isEmpty() ? null : r.get(0);
    }

    public List<User> findAll() {
        return em.createNamedQuery("User.findAll", User.class).getResultList();
    }

    public void updateUser(User user) {
        em.merge(user);
    }

    /** Soft-delete: mark user inactive instead of removing. */
    public void deactivateUser(int userId) {
        User u = findById(userId);
        if (u != null) {
            u.setActive(false);
            em.merge(u);
        }
    }

    /** Reset password and return the new plain-text password. */
    public String resetPassword(int userId, String newPlainPassword) {
        User u = findById(userId);
        if (u == null) throw new IllegalArgumentException("User not found: " + userId);
        u.setPassword(PasswordUtil.hash(newPlainPassword));
        em.merge(u);
        return newPlainPassword;
    }

    public boolean usernameExists(String username) {
        return findByUsername(username) != null;
    }
}
