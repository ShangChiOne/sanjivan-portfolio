package com.crs.entity;

import jakarta.persistence.*;
import java.io.Serializable;
import java.time.LocalDateTime;

@Entity
@Table(name = "email_notifications")
@NamedQuery(name = "EmailNotification.findAll",
            query = "SELECT e FROM EmailNotification e ORDER BY e.sentAt DESC")
public class EmailNotification implements Serializable {

    public enum NotificationType {
        ACCOUNT_CREATED, ACCOUNT_DEACTIVATED, PASSWORD_RESET, RECOVERY_PLAN, MILESTONE_ADDED, PERFORMANCE_REPORT, ELIGIBILITY_RESULT
    }

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "notification_id")
    private Integer notificationId;

    @Column(name = "recipient_email", nullable = false, length = 100)
    private String recipientEmail;

    @Column(name = "subject", nullable = false, length = 255)
    private String subject;

    @Column(name = "body", nullable = false, columnDefinition = "TEXT")
    private String body;

    @Enumerated(EnumType.STRING)
    @Column(name = "notification_type", nullable = false)
    private NotificationType notificationType;

    @Column(name = "is_sent")
    private boolean isSent = true;

    @Column(name = "error_message", columnDefinition = "TEXT")
    private String errorMessage;

    @Column(name = "sent_at", updatable = false)
    private LocalDateTime sentAt;

    @PrePersist
    protected void onCreate() { sentAt = LocalDateTime.now(); }

    // ---- Getters & Setters ----

    public Integer getNotificationId()                      { return notificationId; }
    public void setNotificationId(Integer id)               { this.notificationId = id; }
    public String getRecipientEmail()                       { return recipientEmail; }
    public void setRecipientEmail(String email)             { this.recipientEmail = email; }
    public String getSubject()                              { return subject; }
    public void setSubject(String subject)                  { this.subject = subject; }
    public String getBody()                                 { return body; }
    public void setBody(String body)                        { this.body = body; }
    public NotificationType getNotificationType()           { return notificationType; }
    public void setNotificationType(NotificationType t)     { this.notificationType = t; }
    public boolean isSent()                                 { return isSent; }
    public void setSent(boolean sent)                       { isSent = sent; }
    public String getErrorMessage()                         { return errorMessage; }
    public void setErrorMessage(String msg)                 { this.errorMessage = msg; }
    public LocalDateTime getSentAt()                        { return sentAt; }
}
