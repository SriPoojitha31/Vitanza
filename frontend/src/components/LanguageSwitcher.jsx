import React, { useContext } from "react";
import { I18nContext } from "../i18n/I18nProvider";

export default function LanguageSwitcher() {
  const { lang, setLang, t } = useContext(I18nContext);
  
  const languages = [
    { code: 'en', name: 'English', flag: '🇺🇸' },
    { code: 'hi', name: 'हिन्दी', flag: '🇮🇳' },
    { code: 'te', name: 'తెలుగు', flag: '🇮🇳' },
    { code: 'bn', name: 'বাংলা', flag: '🇧🇩' },
    { code: 'as', name: 'অসমীয়া', flag: '🇮🇳' }
  ];

  return (
    <select 
      value={lang} 
      onChange={(e) => setLang(e.target.value)} 
      style={{ 
        borderRadius: 8, 
        padding: "0.5rem", 
        border: "1px solid #d1d5db",
        backgroundColor: "white",
        fontSize: "0.875rem",
        cursor: "pointer",
        minWidth: "120px"
      }}
      title={t('language')}
    >
      {languages.map(language => (
        <option key={language.code} value={language.code}>
          {language.flag} {language.name}
        </option>
      ))}
    </select>
  );
}

