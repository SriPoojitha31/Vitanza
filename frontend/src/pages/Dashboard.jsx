import React from "react";
import { colors } from "../theme";

export default function Dashboard() {
  return (
    <section style={{ padding: "2rem" }}>
      <h1 style={{ color: colors.primary }}>Welcome to Vitanza</h1>
      <p style={{ color: colors.text }}>
        Smart Health Surveillance & Early Warning System
      </p>
      {/* Add dashboard widgets/components here */}
    </section>
  );
}