/**
 * 2020.02.15: docsify-bibtex.js 与 docsify-footer-enh 冲突
 */

// refer to: 
// 1. https://github.com/pcooksey/bibtex-js/wiki
// 2. https://github.com/pcooksey/bibtex-js/wiki/Customizable-Template
// 3. http://www.cs.cmu.edu/~mmv/Veloso.html

var compile = function (functionObject) {
	return functionObject.toString().match(/\/\*([\s\S]*?)\*\//)[1];
};
var default_bibtex_template = compile(function () {/*
	<div id="bibtex_display">
		<div class="if bibtex_template" style="display: none;">
			<ul><li>
				<!-- if want to get the attr `BIBTEXKEY`, add the class `bibtexVar` -->
				<div class="bibtexVar ref-id" id="+BIBTEXKEY+" extra="BIBTEXKEY"></div>
				<div>
					<span class="sort author" extra="DESC string"></span>
					<span style="float: right;display: none;"><em>@<span class="bibtexkey"></span></em></span>
				</div>
				<div>
					<strong><span class="title"></span></strong>,
					<span class="if journal"><em><span class="journal"></span></em>,</span>
					<span class="if booktitle">In <em><span class="booktitle"></span></em>,</span>
					<span class="if editor"><span class="editor"></span> (editors),</span>
					<span class="if publisher"><em><span class="publisher"></span></em>,</span>
					<span class="if institution"><span class="institution"></span>,</span>
					<span class="if address"><span class="address"></span>,</span>
					<span class="if volume"><span class="volume"></span>,</span>
					<span class="if journal number">(<span class="number"></span>),</span>
					<span class="if pages"> pages <span class="pages"></span>,</span>
					<span class="if month"><span class="month"></span>,</span>
					<span class="if year"><span class="year"></span>.</span>
					<span class="if note"><span class="note"></span>.</span>
				</div>
				<div style="display:none"><span class="bibtextype"></span></div>
				<div style="display:none"><span class="if topic"><span class="topic"></span></span></div>
			</li></ul>
		</div>
	</div>
 */
});
var default_bibtex_structure = compile(function () {/*
	<div class="bibtex_structure">
		<div class="sort year" extra="DESC number">
			<div class="templates"></div>
		</div>
	</div>
 */
});

		
function renderPage(){
	var templates = document.getElementsByClassName('bibtex_template')
	if (templates.length > 0) {
		templates[0].addEventListener("DOMNodeRemoved", function() {
			if ($('#bibtex_display .bibtexentry').length > 0) {
				// 1. 获取参考文献信息
				var ref_map = {}
				$('#bibtex_display li').each(function() {
					var ref_id = $(this).children('.ref-id').attr('id')
					var authors_div = $(this).find('.author')
					// 1.1 获取参考文献的所有作者名
					var authors_span = authors_div.children('span[class!=bibtex_js_conjunction]')
					// 获取参考文献的作者数量
					var num_authors = authors_span.length
					var authors = ''
					// 1.2 格式化作者名
					if (num_authors >= 2) {
						var last_name = $(authors_span[0]).children('.last').text().replace(/^ | $/, '')
						authors += (last_name)
						// 加上“et al.”
						authors += ' et al.'
					} else {
						authors_span.each(function(i) {
							var last_name = $(this).children('.last')
							if (last_name.length > 0) {
								authors += (last_name.text().replace(/^ | $/, '') + ', ')
							}
						})
						// 移除字符串“, ”
						authors = authors.slice(0, -2)
						if(num_authors == 2){
							// 将最后一个“,”替换为“and”
							authors = authors.substring(0, authors.lastIndexOf(',')) + ' and' + authors.substring(authors.lastIndexOf(',') + 1)
						}
					}
					// 1.3 作者名 + 发表年份
					ref_map[ref_id] = authors + ', ' + $(this).find('.year').text()
				})
				// 2. 将文章中的 `[@xxx ...]` 替换为格式化后的“作者名 + 发表年份”
				if (!ref_map.hasOwnProperty('+BIBTEXKEY+')) {
					$('p').each(function() {
						para = $(this).html()
						// 获取格式为“[@xxx] | [@xxx; @xxx] | [@xxx; @xxx; @xxx]”的所有片段，如果不存在，则跳过。
						// var paper_refs = para.match(/\[@.*?\]/g)            // version 1
						// var paper_refs = para.match(/\[?@[a-zA-Z0-9]+\]?/g) // version 2
						var paper_refs = para.match(/\[?@[a-zA-Z0-9-_]+\]?/g)
						var num_ref_code_block = paper_refs != null ? paper_refs.length : 0
						for (var i = 0; i < num_ref_code_block; i++) {
							var paper_ref = paper_refs[i]
							console.log(paper_ref)
							// 获取该片段中所有论文的 id
							// var ref_fmt_match = paper_ref.match(/^\[@(.*?);| @(.*?);|@(.*?)\]$/g)
							var ref_fmt_match = paper_ref.match(/^\[?@([a-zA-Z0-9-_]+);| @([a-zA-Z0-9-_]+);|@([a-zA-Z0-9-_]+)\]?$/g)
							// 将所有 id 循环替换为“作者名 + 发表年份”
							var _paper_ref = paper_ref
							for (var j = 0; j < ref_fmt_match.length; j++) {
								var ref_id = ref_fmt_match[j].replace('@', '').replace('[', '').replace(']', '').replace(' ', '').replace(';', '')
								_paper_ref = _paper_ref.replace('@' + ref_id, ref_map[ref_id]).replace('[', '(').replace(']', ')')
								para = para.replace(paper_ref, _paper_ref)
							}
						}
						$(this).html(para)
					})
				}
			}
		});
	}
}

(function () {
	function install(hook, vm) {
		hook.afterEach(function(html, next) {
			// add bibtex-template
			var dbt = document.getElementById("docsify-bibtex-template")
			if (dbt != null) {
				html += dbt.innerHTML
			}else{
				html += default_bibtex_template
			}
			
			// add bibtex-structure
			if($docsify.bibtex.sort){
				var dbs = document.getElementById("docsify-bibtex-structure")
				if (dbs != null) {
					html += dbs.innerHTML
				}else{
					html += default_bibtex_structure
				}
			}
			next(html)
		});
		hook.ready(function() {
			renderPage()
		});
	}
	$docsify.plugins = [].concat(install, $docsify.plugins);
}());