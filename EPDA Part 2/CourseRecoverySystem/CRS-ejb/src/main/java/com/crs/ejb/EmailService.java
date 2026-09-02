package com.crs.ejb;

import com.crs.entity.EmailNotification;

import jakarta.annotation.Resource;
import jakarta.ejb.Asynchronous;
import jakarta.ejb.Stateless;
import jakarta.mail.Message;
import jakarta.mail.MessagingException;
import jakarta.mail.Session;
import jakarta.mail.Transport;
import jakarta.mail.internet.InternetAddress;
import jakarta.mail.internet.MimeMessage;
import jakarta.persistence.EntityManager;
import jakarta.persistence.PersistenceContext;

/**
 * Sends email notifications and logs them to the database.
 *
 * The mail/CRSMailSession resource must be configured in GlassFish
 * (Admin Console → Resources → JavaMail Sessions).
 */
@Stateless
public class EmailService {

    @PersistenceContext(unitName = "CRSPU")
    private EntityManager em;

    @Resource(lookup = "mail/CRSMailSession")
    private Session mailSession;

    @Asynchronous
    public void sendAccountCreated(String recipientEmail, String recipientName, String tempPassword) {
        String subject = "CRS – Your Account Has Been Created";
        String body = "Dear " + recipientName + ",\n\n"
                + "Your Course Recovery System account has been created.\n"
                + "Username:        " + recipientEmail + "\n"
                + "Temp Password:   " + tempPassword + "\n\n"
                + "Please log in and change your password immediately.\n\n"
                + "Regards,\nCRS Administration";
        send(recipientEmail, subject, body, EmailNotification.NotificationType.ACCOUNT_CREATED);
    }

    @Asynchronous
    public void sendAccountDeactivated(String recipientEmail, String recipientName) {
        String subject = "CRS – Account Deactivated";
        String body = "Dear " + recipientName + ",\n\n"
                + "Your Course Recovery System account has been deactivated.\n"
                + "Please contact your administrator if you believe this is an error.\n\n"
                + "Regards,\nCRS Administration";
        send(recipientEmail, subject, body, EmailNotification.NotificationType.ACCOUNT_DEACTIVATED);
    }

    @Asynchronous
    public void sendPasswordReset(String recipientEmail, String recipientName, String newPassword) {
        String subject = "CRS – Password Reset";
        String body = "Dear " + recipientName + ",\n\n"
                + "Your password has been reset.\n"
                + "New Password: " + newPassword + "\n\n"
                + "Regards,\nCRS Administration";
        send(recipientEmail, subject, body, EmailNotification.NotificationType.PASSWORD_RESET);
    }

    @Asynchronous
    public void sendRecoveryPlanNotification(String recipientEmail, String studentName,
                                             String courseName) {
        String subject = "CRS – Course Recovery Plan Created";
        String body = "Dear " + studentName + ",\n\n"
                + "A recovery plan for [" + courseName + "] has been created for you.\n"
                + "Please log in to CRS to view your milestones and action plan.\n\n"
                + "Regards,\nCRS Administration";
        send(recipientEmail, subject, body, EmailNotification.NotificationType.RECOVERY_PLAN);
    }

    @Asynchronous
    public void sendMilestoneAdded(String recipientEmail, String studentName,
                                   String studyWeek, String taskDescription) {
        String subject = "CRS – New Milestone Added to Your Recovery Plan";
        String body = "Dear " + studentName + ",\n\n"
                + "A new milestone has been added to your course recovery plan:\n\n"
                + "  Study Week:  " + studyWeek + "\n"
                + "  Task:        " + taskDescription + "\n\n"
                + "Please log in to CRS to view the full details and due date.\n\n"
                + "Regards,\nCRS Administration";
        send(recipientEmail, subject, body, EmailNotification.NotificationType.MILESTONE_ADDED);
    }

    @Asynchronous
    public void sendPerformanceReport(String recipientEmail, String studentName,
                                      int semester, int year) {
        String subject = "CRS – Academic Performance Report (Sem " + semester + ", Year " + year + ")";
        String body = "Dear " + studentName + ",\n\n"
                + "Your academic performance report for Semester " + semester
                + ", Year " + year + " is now available in CRS.\n\n"
                + "Regards,\nCRS Administration";
        send(recipientEmail, subject, body, EmailNotification.NotificationType.PERFORMANCE_REPORT);
    }

    @Asynchronous
    public void sendEligibilityResult(String recipientEmail, String studentName, boolean eligible) {
        String subject = "CRS – Eligibility Check Result";
        String body = "Dear " + studentName + ",\n\n"
                + (eligible
                   ? "Congratulations! You are ELIGIBLE to progress to the next level of study."
                   : "Unfortunately, you are NOT eligible to progress at this time. "
                     + "Please consult your Academic Officer for a course recovery plan.")
                + "\n\nRegards,\nCRS Administration";
        send(recipientEmail, subject, body, EmailNotification.NotificationType.ELIGIBILITY_RESULT);
    }

    // ---- Private helper ----

    private void send(String to, String subject, String body,
                      EmailNotification.NotificationType type) {
        EmailNotification log = new EmailNotification();
        log.setRecipientEmail(to);
        log.setSubject(subject);
        log.setBody(body);
        log.setNotificationType(type);

        try {
            MimeMessage msg = new MimeMessage(mailSession);
            msg.setFrom(new InternetAddress(mailSession.getProperty("mail.from"), "CRS System"));
            msg.setRecipients(Message.RecipientType.TO, InternetAddress.parse(to));
            msg.setSubject(subject);
            msg.setText(body);
            Transport.send(msg);
            log.setSent(true);
        } catch (Exception e) {
            log.setSent(false);
            log.setErrorMessage(e.getMessage());
        } finally {
            em.persist(log);
        }
    }
}
