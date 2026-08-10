-- Render small-font blocks in the LaTeX preview build only.
--   * a ::: tablenote ::: div: table captions keep their first sentence above
--     the table; the explanatory detail moves into one of these notes below.
--   * a figure-caption paragraph: any paragraph whose text begins "Figure N:"
--     (the draft's caption convention, sitting below its image) is shrunk to
--     match, so figure descriptions read smaller than body text.
function Div(el)
  if el.classes:includes("tablenote") then
    table.insert(el.content, 1,
                 pandoc.RawBlock("latex", "\\begingroup\\footnotesize"))
    table.insert(el.content, pandoc.RawBlock("latex", "\\endgroup"))
    return el
  end
end

function Para(el)
  local c = el.content
  if #c >= 3 and c[1].t == "Str" and c[1].text == "Figure"
     and c[2].t == "Space" and c[3].t == "Str"
     and c[3].text:match("^%d+:$") then
    return {
      pandoc.RawBlock("latex", "\\begingroup\\footnotesize"),
      el,
      pandoc.RawBlock("latex", "\\endgroup"),
    }
  end
end
