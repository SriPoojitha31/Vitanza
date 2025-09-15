import React from "react";
import { colors } from "../theme";

export default function Footer() {
  return (
    <footer style={{
      background: colors.primary,
      color: colors.background,
      textAlign: "center",
      padding: "1rem",
      marginTop: "2rem"
    }}>
      &copy; {new Date().getFullYear()} Vitanza. All rights reserved.
    </footer>
  );
}