import React, { useContext } from "react";
import { I18nContext } from "../i18n/I18nProvider";

export default function LanguageSwitcher() {
  const { lang, setLang } = useContext(I18nContext);
  return (
    <select value={lang} onChange={(e) => setLang(e.target.value)} style={{ borderRadius: 8, padding: "0.25rem 0.5rem" }}>
      <option value="en">English</option>
      <option value="hi">हिंदी</option>
      <option value="te">తెలుగు</option>
      <option value="bn">বাংলা</option>
      <option value="as">অসমীয়া</option>
    </select>
  );
}

