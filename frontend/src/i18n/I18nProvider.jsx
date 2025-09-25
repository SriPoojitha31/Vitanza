import React, { useEffect, useState } from "react";
import as from "../locales/as.json";
import bn from "../locales/bn.json";
import en from "../locales/en.json";
import hi from "../locales/hi.json";
import te from "../locales/te.json";

const dictionaries = { en, hi, te, bn, as };

export const I18nContext = React.createContext({ t: (k) => k, lang: "en", setLang: () => {} });

export function I18nProvider({ children }) {
  const [lang, setLang] = useState(localStorage.getItem("lang") || "en");
  useEffect(() => { localStorage.setItem("lang", lang); }, [lang]);
  const t = (key) => {
    const dict = dictionaries[lang] || {};
    return key.split('.').reduce((obj, part) => (obj && obj[part] !== undefined ? obj[part] : undefined), dict) ?? key;
  };
  return (
    <I18nContext.Provider value={{ t, lang, setLang }}>
      {children}
    </I18nContext.Provider>
  );
}
