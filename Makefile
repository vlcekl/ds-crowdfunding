# Make documentation

proposal.html: proposal.md proposal.css
    pandoc -o $@ -f markdown -t html -c proposal.css proposal.md

proposal.pdf: proposal.html
    wkhtmltopdf proposal.html $@
