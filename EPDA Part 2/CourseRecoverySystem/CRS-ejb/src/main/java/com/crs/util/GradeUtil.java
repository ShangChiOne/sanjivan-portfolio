package com.crs.util;

/**
 * Utility for converting between letter grades and grade points (4.0 scale).
 */
public class GradeUtil {

    private GradeUtil() {}

    public static double toGradePoint(String grade) {
        if (grade == null) return 0.0;
        return switch (grade.trim().toUpperCase()) {
            case "A+" -> 4.0;
            case "A"  -> 4.0;
            case "A-" -> 3.7;
            case "B+" -> 3.3;
            case "B"  -> 3.0;
            case "B-" -> 2.7;
            case "C+" -> 2.3;
            case "C"  -> 2.0;
            case "C-" -> 1.7;
            case "D+" -> 1.3;
            case "D"  -> 1.0;
            default   -> 0.0;  // F or unrecognised
        };
    }

    /** A course is passed when grade point >= 2.0 (grade C or above). */
    public static boolean isPassed(String grade) {
        return toGradePoint(grade) >= 2.0;
    }

    /**
     * CGPA = Σ(gradePoint × creditHours) / Σ(creditHours)
     *
     * @param gradePoints   parallel array of grade points for each course
     * @param creditHours   parallel array of credit hours for each course
     * @return CGPA rounded to 2 decimal places
     */
    public static double calculateCGPA(double[] gradePoints, int[] creditHours) {
        double totalPoints = 0;
        int totalHours = 0;
        for (int i = 0; i < gradePoints.length; i++) {
            totalPoints += gradePoints[i] * creditHours[i];
            totalHours  += creditHours[i];
        }
        if (totalHours == 0) return 0.0;
        return Math.round((totalPoints / totalHours) * 100.0) / 100.0;
    }
}
